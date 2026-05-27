import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from models.transformer import TransformerEncoder, TransformerDecoder
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch import amp
from utils.paths import PROJECT_ROOT

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

class MLMHead(nn.Module):
    def __init__(self, n_embd, vocab_size):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, n_embd)
        self.activation = nn.GELU()
        self.ln = nn.LayerNorm(n_embd)
        self.linear2 = nn.Linear(n_embd, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.linear2.bias = self.bias

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.ln(x)
        logits = self.linear2(x)
        return logits


class FeatureProjector(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.net(x)


class IFIM(nn.Module):
    def __init__(self, model_configs, c=None, p_ci=None):
        super().__init__()
        interaction_decoder_config = model_configs.get(
            'InteractionDecoder',
            model_configs.get('TransformerDeocder')
        )
        if interaction_decoder_config is None:
            raise KeyError("model_config.yaml must define InteractionDecoder")

        # Params
        d_model = model_configs['DrugEncoder']['d_model']
        n_heads = model_configs['DrugEncoder']['n_head']
        vocab_size = model_configs['DrugEncoder']['vocab_size']
        fusion_n_heads = interaction_decoder_config['n_head']
        self.d_model = d_model
        self.encoder_n_heads = n_heads
        self.fusion_n_heads = fusion_n_heads
        d_esm = 1280


   
        self.use_llm = model_configs.get('use_llm', True)
        llm_model_path = Path(model_configs.get('llm_model_path', "Qwen"))
        self.llm_model_path = str(llm_model_path if llm_model_path.is_absolute() else PROJECT_ROOT / llm_model_path)

        # Drug encoder
        self.precompute_freqs_cis = precompute_freqs_cis(d_model // n_heads, 4000)
        self.drug_encoder = TransformerEncoder(config=model_configs['DrugEncoder'])
        self.MLMHead = MLMHead(d_model, vocab_size)

        # Fusion
        self.aggregator = TransformerDecoder(config=interaction_decoder_config)
        self.aggregator_protein = TransformerDecoder(config=interaction_decoder_config)
        self.classifier = nn.Linear(2 * (self.d_model // self.fusion_n_heads), 2)


        # interventional training
        self.c = c
        self.p_ci = p_ci
        self.c_center = c.size(0)
        self.ln = nn.LayerNorm(d_esm)

        if (1280 % self.c_center) != 0:
            raise RuntimeError(F"d_model % self.c_center)!=0")
        elif self.c_center != fusion_n_heads:
            raise RuntimeError(F"confounder dict self.c_center != n_heads")
        else:
            dim = int(d_esm / self.c_center)

        self.linear_q = nn.Linear(d_esm, d_esm)
        self.linear_k = nn.Linear(d_esm, dim)
        self.linear_v = nn.Linear(d_esm, dim)
        self.pr_linear = nn.Linear(d_esm, d_model)

        self.mlp_fusion = torch.nn.Sequential(
            torch.nn.Linear(self.d_model * 2, self.d_model),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.d_model, self.d_model)
        )

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_path,
            trust_remote_code=True
        )

        self.special_tokens = [
            "<drug_emb>",
            "<protein_emb>",
            "<adjusted_prot_emb>",
            "<adjusted_drug_emb>",
        ]

        self.llm_tokenizer.add_tokens(self.special_tokens, special_tokens=True)

        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.protein_token_id = self.llm_tokenizer.convert_tokens_to_ids("<protein_emb>")
        self.drug_token_id = self.llm_tokenizer.convert_tokens_to_ids("<drug_emb>")
        self.adjusted_prot_token_id = self.llm_tokenizer.convert_tokens_to_ids("<adjusted_prot_emb>")
        self.adjusted_drug_token_id = self.llm_tokenizer.convert_tokens_to_ids("<adjusted_drug_emb>")

        print(f"Protein token ID: {self.protein_token_id}")
        print(f"Drug token ID: {self.drug_token_id}")
        print(f"adjusted prot token ID: {self.adjusted_prot_token_id }")
        print(f"adjusted drug token ID: {self.adjusted_drug_token_id}")

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        self.llm.resize_token_embeddings(len(self.llm_tokenizer))

        lora_config = LoraConfig(
            r=model_configs.get('lora_r', 8),
            lora_alpha=model_configs.get('lora_alpha', 16),
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],

            lora_dropout=model_configs.get('lora_dropout', 0.1),
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()

        self.llm_hidden_size = self.llm.config.hidden_size

        self.protein_projector = FeatureProjector(
            in_dim=d_model, 
            out_dim=self.llm_hidden_size,
            hidden_dim=512
        )
        self.drug_projector = FeatureProjector(
            in_dim=d_model,  
            out_dim=self.llm_hidden_size,
            hidden_dim=512
        )

        self.prompt_template = [
            "Input: "
            "Drug representation embedding: <drug_emb>. "
            "Protein representation embedding: <protein_emb>. "
    "Context: "
            "Based on induced-fit theory, binding may involve dynamic conformational adjustments on protein side. "
            "Reason about mutual adaptation, flexibility, and stabilizing interaction patterns implied by both drug and protein representation embeddings. "
    "Output: "
            "Produce adjusted protein representation embedding <adjusted_prot_emb> and adjusted drug representation embedding <adjusted_drug_emb> according to the input and induced-fit theory mentioned in context."
        ]

        self.protein_extractor = nn.Sequential(
            nn.Linear(self.llm_hidden_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        )
        self.drug_extractor = nn.Sequential(
            nn.Linear(self.llm_hidden_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        )


        self.embed_tokens = self.llm.get_input_embeddings()


    def encode_drug(self, input_drugs, freqs_cis):
        drug_id = input_drugs['input_ids']

        drug_padding_mask = ~input_drugs['attention_mask'].bool()
        bz, len_d = drug_id.size()
        drug_attn_mask = drug_padding_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.encoder_n_heads, len_d, -1)
        drug_f = self.drug_encoder(drug_id, drug_attn_mask, freqs_cis[:len_d, :].to(drug_id.device))
        return drug_f

    def encode_protein(self, pr_f, pr_mask):
        bz = pr_mask.size(0)
        c_i = self.c.unsqueeze(0).expand(bz, -1, -1)
        pr_f = self.confounder_alignment_module(pr_f, c_i)
        return pr_f, pr_mask

    def confounder_alignment_module(self, pr_f, ci):
        device = pr_f.device
        bz = pr_f.size(0)
        Q = self.linear_q(pr_f).view(bz, -1, self.c_center, 1280 // self.c_center).permute(0, 2, 1, 3)
        K = self.linear_k(ci).unsqueeze(1).permute(0, 2, 1, 3)
        V = self.linear_v(ci).unsqueeze(1).permute(0, 2, 1, 3)
        A = torch.matmul(Q, K.permute(0, 1, 3, 2))
        A = F.softmax(A / torch.sqrt(torch.tensor(K.shape[1], dtype=torch.float32, device=device)), dim=-1)
        Q = torch.matmul(A, V)
        Q = Q.permute(0, 2, 1, 3).contiguous()
        Q = Q.view(bz, -1, 1280)
        Q = Q + pr_f
        Q = self.pr_linear(Q)
        return Q


    def llm_interaction_with_prompt(self, protein_emb, drug_emb):
      
        batch_size = protein_emb.size(0)
        device = protein_emb.device

        prompts = [self.prompt_template[0]] * batch_size

        encodings = self.llm_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        inputs_embeds = self.embed_tokens(input_ids)

        protein_proj = self.protein_projector(protein_emb)  
        drug_proj = self.drug_projector(drug_emb)  

        for i in range(batch_size):
            drug_pos = (input_ids[i] == self.drug_token_id).nonzero(as_tuple=True)[0].item()
            protein_pos = (input_ids[i] == self.protein_token_id).nonzero(as_tuple=True)[0].item()

            inputs_embeds[i, drug_pos] = drug_proj[i, 0]
            inputs_embeds[i, protein_pos] = protein_proj[i, 0]

        with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        hidden_states = outputs.hidden_states[-1]  


        adjusted_drug_hidden_list = []
        adjusted_prot_hidden_list = []

        for i in range(batch_size):
            adjusted_drug_pos = (input_ids[i] == self.adjusted_drug_token_id).nonzero(as_tuple=True)[0].item()
            adjusted_prot_pos = (input_ids[i] == self.adjusted_prot_token_id).nonzero(as_tuple=True)[0].item()
            adjusted_drug_hidden = hidden_states[i, adjusted_drug_pos]  
            adjusted_drug_hidden_list.append(adjusted_drug_hidden)
            adjusted_prot_hidden = hidden_states[i, adjusted_prot_pos] 
            adjusted_prot_hidden_list.append(adjusted_prot_hidden)

        adjusted_drug_hidden = torch.stack(adjusted_drug_hidden_list, dim=0).unsqueeze(1) 
        adjusted_drug_hidden = adjusted_drug_hidden.float()
        adjusted_prot_hidden = torch.stack(adjusted_prot_hidden_list, dim=0).unsqueeze(1) 
        adjusted_prot_hidden = adjusted_prot_hidden.float()

        new_protein = self.protein_extractor(adjusted_prot_hidden)
        new_drug = self.drug_extractor(adjusted_drug_hidden)

        return new_protein, new_drug

    def fusion(self, drug_f, pr_f, drug_padding_mask, protein_padding_mask):
        bz, len_d, _ = drug_f.size()
        drug_attn_mask = ~drug_padding_mask.bool().unsqueeze(1).unsqueeze(1).expand(-1, self.fusion_n_heads, len_d, -1)
        cross_attn_mask = ~protein_padding_mask.bool().unsqueeze(1).unsqueeze(1).expand(-1, self.fusion_n_heads, len_d,
                                                                                        -1)
        fusion_f, attention_map = self.aggregator(
            src=pr_f,
            tgt=drug_f,
            self_attn_mask=drug_attn_mask,
            cross_attn_mask=cross_attn_mask
        )
        return fusion_f, attention_map

    def fusion_protein(self, pr_f, drug_f, protein_padding_mask, drug_padding_mask):
        bz, len_p, _ = pr_f.size()
        protein_attn_mask = ~protein_padding_mask.bool().unsqueeze(1).unsqueeze(1).expand(-1, self.fusion_n_heads, len_p, -1)
        cross_attn_mask = ~drug_padding_mask.bool().unsqueeze(1).unsqueeze(1).expand(-1, self.fusion_n_heads, len_p, -1)
        fusion_f, attention_map = self.aggregator_protein(
            src=drug_f,
            tgt=pr_f,
            self_attn_mask=protein_attn_mask,
            cross_attn_mask=cross_attn_mask
        )
        return fusion_f, attention_map



    def backdoor_adjustment(self, logits):
        p_ci = self.p_ci.unsqueeze(0).unsqueeze(-1)
        logits = logits * p_ci
        return logits.sum(1)


    def forward(self, input_drugs, input_proteins, pr_mask=None, masked_drugs=None):
        drug_f = self.encode_drug(input_drugs, self.precompute_freqs_cis)  # [B, Ld, 256]
        pr_f, pr_mask = self.encode_protein(input_proteins, pr_mask)  # [B, Lp, 256]


        drug_global = (drug_f * input_drugs['attention_mask'].unsqueeze(-1).float()).sum(1) / \
                      input_drugs['attention_mask'].sum(1, keepdim=True).clamp(min=1)

        protein_global = (pr_f * pr_mask.unsqueeze(-1).float()).sum(1) / \
                         pr_mask.sum(1, keepdim=True).clamp(min=1)

        new_protein_token, new_drug_token = self.llm_interaction_with_prompt(
            protein_global.unsqueeze(1), drug_global.unsqueeze(1)
        )

        drug_f = torch.cat([new_drug_token, drug_f], dim=1)
        pr_f = torch.cat([new_protein_token, pr_f], dim=1)

        batch_size = drug_f.size(0)
        device = drug_f.device

        new_drug_mask = torch.ones((batch_size, 1), device=device)
        drug_attn_mask = torch.cat([new_drug_mask, input_drugs['attention_mask']], dim=1)

        new_pr_mask = torch.ones((batch_size, 1), device=device)
        pr_mask_extended = torch.cat([new_pr_mask, pr_mask], dim=1)


        fusion_drug, attn_map = self.fusion(drug_f, pr_f, drug_attn_mask, pr_mask_extended)
        fusion_protein, attn_map_protein = self.fusion_protein(pr_f, drug_f, pr_mask_extended, drug_attn_mask)
        fusion_f = torch.cat([fusion_drug, fusion_protein], dim=1)


        drug_mlm_logits = None
        if masked_drugs is not None:
            drug_f_mlm = self.encode_drug(masked_drugs, self.precompute_freqs_cis)
            drug_mlm_logits = self.MLMHead(drug_f_mlm).permute(0, 2, 1)

        fusion_final = torch.cat([
            fusion_drug.mean(dim=1),
            fusion_protein.mean(dim=1)
        ], dim=-1)
        fusion_final = fusion_final.view(
            fusion_final.size(0),
            self.c_center,
            2 * (self.d_model // self.c_center)
        )
        c_i = fusion_final
        logits = self.classifier(c_i)  # (B, 2)
        logits = self.backdoor_adjustment(logits)

        return {
            'logits': logits,
            'fusion_f': fusion_f,
            'attn_map': attn_map,
            'attn_map_protein': attn_map_protein,
            'drug_mlm_logits': drug_mlm_logits,
            'drug_cond': fusion_drug,
            'protein_cond': fusion_protein
        }
