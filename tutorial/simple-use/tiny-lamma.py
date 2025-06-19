from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 🎯 统一定义设备
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ✅ 加载并迁移模型
model = AutoLigerKernelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": device}  # ⚠️ 关键：强制模型主设备
).eval()

# ✅ 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ 迁移输入数据（覆盖所有张量）
input_text = "用通俗的中文解释一下量子纠缠"
inputs = tokenizer(input_text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}  # 🚀 字典内全量迁移

# ✅ 生成时传递设备参数
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,      
    )

print("\n🚀 模型生成输出：\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

