import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mlpe.architectures.flows.iaf import InverseAutoRegressiveFlow
from mlpe.architectures.embeddings import ResNet
from mlpe.injection.priors import nonspin_bbh_chirp_mass_q_parameter_sampler

# 定义相关的参数
inference_params = [
    "chirp_mass", "mass_ratio", "luminosity_distance", "phase", 
    "theta_jn", "dec", "psi", "phi"
]
ifos = ["H1", "L1"]
sample_rate = 2048
time_duration = 4

# 计算模型维度
param_dim = len(inference_params)
n_ifos = len(ifos)
strain_dim = int(sample_rate * time_duration)

# 定义嵌入网络（ResNet）
embedding = ResNet(
    (n_ifos, strain_dim),
    context_dim=100,  # 假设 context_dim 为 100，你可以根据自己的定义调整
    layers=[5, 5],    # 假设你使用 5 层 ResNet
    norm_groups=8     # 假设 norm_groups 为 8
)

# 创建教师模型
teacher_model = InverseAutoRegressiveFlow(
    (param_dim, n_ifos, strain_dim),
    embedding,
    torch.optim.AdamW,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    inference_params,
    nonspin_bbh_chirp_mass_q_parameter_sampler,
    num_transforms=50,  # 教师模型的参数
    num_blocks=4,
    hidden_features=100
)

# 创建学生模型
student_model = InverseAutoRegressiveFlow(
    (param_dim, n_ifos, strain_dim),
    embedding,
    torch.optim.AdamW,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    inference_params,
    nonspin_bbh_chirp_mass_q_parameter_sampler,
    num_transforms=30,  # 学生模型的参数
    num_blocks=4,
    hidden_features=80
)

def jsd(P, Q, eps=1e-10):
    P = P + eps
    Q = Q + eps
    M = 0.5 * (P + Q)
    kl_pm = F.kl_div(torch.log(P), M, reduction='batchmean')
    kl_qm = F.kl_div(torch.log(Q), M, reduction='batchmean')
    jsd_value = 0.5 * (kl_pm + kl_qm)
    return jsd_value

# 比较模型权重的JSD
def compare_model_weights_jsd(teacher_model, student_model):
    """
    比较教师模型和学生模型的所有权重，并计算它们的 Jensen-Shannon Divergence (JSD)。
    
    teacher_model: 已训练好的教师模型
    student_model: 学生模型
    """
    jsd_values = []
    
    for (teacher_param, student_param) in zip(teacher_model.parameters(), student_model.parameters()):
        # 检查权重张量的形状是否匹配
        if teacher_param.shape != student_param.shape:
            print(f"跳过形状不匹配的层: 教师模型层大小 {teacher_param.shape}, 学生模型层大小 {student_param.shape}")
            continue  # 跳过形状不匹配的层
        
        # 将模型权重转换为概率分布（通过归一化）
        teacher_weights = teacher_param.data
        student_weights = student_param.data
        
        # 将权重展平为一维向量
        teacher_weights = teacher_weights.view(-1)
        student_weights = student_weights.view(-1)
        
        # 将权重归一化为概率分布
        teacher_probs = F.softmax(teacher_weights, dim=0)
        student_probs = F.softmax(student_weights, dim=0)
        
        # 计算每层权重的JSD
        jsd_value = jsd(teacher_probs, student_probs)
        jsd_values.append(jsd_value.item())  # 获取数值并存储
        
    return jsd_values
    
# 从之前保存的路径中加载权重
base_dir = os.getenv("BASE_DIR", "/default/path")  # 如果没有环境变量，可以提供默认路径
checkpoint_dir = os.path.join(base_dir, "checkpoints2")

# 拼接权重文件路径
teacher_checkpoint_path = os.path.join(checkpoint_dir, "teacher_model_checkpoint-v500.ckpt")
student_checkpoint_path = os.path.join(checkpoint_dir, "student_model_checkpoint-v500.ckpt")

# 加载教师模型和学生模型权重
teacher_model.load_state_dict(torch.load(teacher_checkpoint_path)['state_dict'])
teacher_model.eval()

student_model.load_state_dict(torch.load(student_checkpoint_path)['state_dict'])
student_model.eval()

jsd_results = compare_model_weights_jsd(teacher_model, student_model)

# 输出每层的JSD结果
for i, jsd_value in enumerate(jsd_results):
    print(f"第 {i + 1} 层的 JSD: {jsd_value:.4f}")

plt.plot(jsd_results)
plt.xlabel('Layer')
plt.ylabel('JSD Value')
plt.title('JSD between Teacher and Student Models')
plt.savefig('JSD_comparsion-v500.png')