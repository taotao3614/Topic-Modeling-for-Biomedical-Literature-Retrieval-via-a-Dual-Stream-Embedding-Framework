import csv
import numpy as np
import os
from tqdm import tqdm
import json
from sentence_transformers import SentenceTransformer
import pandas as pd
import itertools
import sys

def load_cleaned_data(csv_path):
    """
    从清洗后的CSV文件中加载数据
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        df: 包含所有原始数据的DataFrame
        pmids: PMID列表
        texts: 文本列表
        mesh_terms_list: MeSH术语列表（每篇文章的MeSH术语集合）
    """
    print(f"正在加载清洗后的数据: {csv_path}")
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误：文件 {csv_path} 不存在!")
        # 尝试定位文件
        current_dir = os.getcwd()
        print(f"当前工作目录: {current_dir}")
        parent_dir = os.path.dirname(current_dir)
        alternative_path = os.path.join(parent_dir, 'raw.csv')
        if os.path.exists(alternative_path):
            print(f"找到替代路径: {alternative_path}")
            csv_path = alternative_path
        else:
            print(f"在父目录中也未找到 raw.csv")
            sys.exit(1)
    
    # 直接读取整个CSV文件到DataFrame
    df = pd.read_csv(csv_path)
    
    pmids = df['pmid'].tolist()
    texts = df['text'].tolist()
    
    # 解析JSON格式的mesh_terms字段
    mesh_terms_list = []
    for i, mesh_terms_json in enumerate(df['mesh_terms']):
        try:
            if mesh_terms_json and mesh_terms_json != '[]':
                mesh_terms = json.loads(mesh_terms_json)
            else:
                mesh_terms = []
        except json.JSONDecodeError:
            pmid = pmids[i] if i < len(pmids) else "未知"
            print(f"警告: PMID {pmid} 的MeSH术语解析失败，将使用空列表代替")
            mesh_terms = []
        
        mesh_terms_list.append(mesh_terms)
    
    print(f"加载完成，共 {len(texts)} 条记录")
    return df, pmids, texts, mesh_terms_list

def embed_texts(texts, model_name='NeuML/pubmedbert-base-embeddings', batch_size=64):
    """
    使用指定的模型对文本进行嵌入
    
    Args:
        texts: 文本列表
        model_name: 模型名称或路径
        batch_size: 批处理大小
        
    Returns:
        embeddings: 嵌入向量数组
    """
    print(f"正在加载模型: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("开始文本嵌入...")
    embeddings = model.encode(texts, 
                              show_progress_bar=True, 
                              batch_size=batch_size)
    
    print(f"嵌入完成，向量维度: {embeddings.shape}")
    return embeddings

def embed_mesh_terms(mesh_terms_list, beta=0.7, model_name='dmis-lab/biobert-v1.1', batch_size=64):
    """
    对MeSH术语进行嵌入并加权融合，先进行embedding，再加权
    
    Args:
        mesh_terms_list: MeSH术语列表（每篇文章的MeSH术语集合）
        beta: MeSH术语权重参数，用于控制主要主题(MajorTopicYN='Y')的权重
             主要主题权重为beta，次要主题权重为(1-beta)
        model_name: 模型名称或路径
        batch_size: 批处理大小
        
    Returns:
        mesh_embeddings: 加权融合后的嵌入向量数组
        has_mesh: 指示每篇文章是否有MeSH术语的布尔数组
    """
    print(f"正在加载MeSH术语嵌入模型: {model_name}")
    model = SentenceTransformer(model_name)
    
    # 根据beta计算主要主题和次要主题的权重
    weight_major = beta  # MajorTopicYN='Y'的权重
    weight_minor = 1 - beta  # MajorTopicYN='N'的权重
    
    # 准备存储所有文章的MeSH术语嵌入向量
    mesh_embeddings = []
    has_mesh = []  # 记录每篇文章是否有MeSH术语
    
    # 初始化一个变量来存储向量维度
    vector_dim = None
    
    print("开始处理MeSH术语嵌入...")
    print(f"主要主题权重: {weight_major}, 次要主题权重: {weight_minor}")
    
    for mesh_terms in tqdm(mesh_terms_list):
        # 判断是否有有效的MeSH术语
        valid_mesh = bool(mesh_terms)
        has_mesh.append(valid_mesh)
        
        # 如果文章没有MeSH术语，使用零向量
        if not valid_mesh:
            # 如果还不知道向量维度，需要进行一次试验性嵌入
            if vector_dim is None:
                dummy_embedding = model.encode(["dummy"])
                vector_dim = dummy_embedding.shape[1]
            mesh_embeddings.append(np.zeros(vector_dim))
            continue
        
        # 提取所有DescriptorName及其MajorTopicYN值
        descriptor_names = []
        major_flags = []
        
        for term in mesh_terms:
            if 'DescriptorName' in term:
                descriptor_names.append(term['DescriptorName'])
                # 记录是否为主要主题
                is_major = term.get('MajorTopicYN', 'N') == 'Y'
                major_flags.append(is_major)
        
        if not descriptor_names:
            # 如果没有有效的DescriptorName，使用零向量并更新has_mesh
            has_mesh[-1] = False
            if vector_dim is None:
                dummy_embedding = model.encode(["dummy"])
                vector_dim = dummy_embedding.shape[1]
            mesh_embeddings.append(np.zeros(vector_dim))
            continue
        
        # 先对所有MeSH术语进行嵌入
        term_embeddings = model.encode(descriptor_names, batch_size=batch_size)
        if vector_dim is None:
            vector_dim = term_embeddings.shape[1]
        
        # 根据MajorTopicYN值分配权重
        weights = np.array([weight_major if is_major else weight_minor for is_major in major_flags])
        
        # 归一化权重
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        # 加权求和
        weighted_embedding = np.zeros(vector_dim)
        for j, embedding in enumerate(term_embeddings):
            weighted_embedding += embedding * weights[j]
        
        mesh_embeddings.append(weighted_embedding)
    
    mesh_embeddings = np.array(mesh_embeddings)
    has_mesh = np.array(has_mesh)
    print(f"MeSH术语嵌入完成，向量维度: {mesh_embeddings.shape}，有MeSH术语的文章数: {has_mesh.sum()}")
    return mesh_embeddings, has_mesh

def combine_embeddings(text_embeddings, mesh_embeddings, has_mesh, alpha=0.7):
    """
    将文本嵌入和MeSH术语嵌入进行加权平均
    
    Args:
        text_embeddings: 文本嵌入向量数组
        mesh_embeddings: MeSH术语嵌入向量数组
        has_mesh: 指示每篇文章是否有MeSH术语的布尔数组
        alpha: 文本嵌入权重参数，文本嵌入权重为alpha，MeSH术语嵌入权重为(1-alpha)
        
    Returns:
        combined_embeddings: 加权平均后的嵌入向量数组
    """
    print("开始融合文本嵌入和MeSH术语嵌入...")
    
    # 根据alpha计算文本和MeSH术语的权重
    text_weight = alpha
    mesh_weight = 1 - alpha
    print(f"文本嵌入权重: {text_weight}, MeSH术语嵌入权重: {mesh_weight}")
    
    # 确保输入数组的形状相同
    assert text_embeddings.shape[0] == mesh_embeddings.shape[0] == has_mesh.shape[0], "输入数组的长度不一致"
    
    # 创建结果数组
    combined_embeddings = np.zeros_like(text_embeddings)
    
    # 对于有MeSH术语的文章，应用加权平均
    mask_has_mesh = has_mesh
    
    # 有MeSH术语的文章：加权平均
    combined_embeddings[mask_has_mesh] = (
        text_weight * text_embeddings[mask_has_mesh] + 
        mesh_weight * mesh_embeddings[mask_has_mesh]
    )
    
    # 没有MeSH术语的文章：仅使用文本嵌入
    combined_embeddings[~mask_has_mesh] = text_embeddings[~mask_has_mesh]
    
    print(f"嵌入融合完成，向量维度: {combined_embeddings.shape}")
    return combined_embeddings

def save_for_milvus(output_dir, output_prefix, df, embeddings):
    """
    保存适合Milvus导入的数据文件
    
    Args:
        output_dir: 输出目录
        output_prefix: 输出文件前缀
        df: 原始数据DataFrame
        embeddings: 嵌入向量数组
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 只保留需要的列：pmid, title, pub_year
    milvus_data = df[['pmid', 'title', 'pub_year']].copy()
    
    # 保存为npz文件，同时包含向量和元数据
    output_path = os.path.join(output_dir, f"{output_prefix}_milvus_data.npz")
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        pmids=milvus_data['pmid'].values,
        titles=milvus_data['title'].values,
        pub_years=milvus_data['pub_year'].values
    )
    print(f"Milvus数据已保存至: {output_path}")
    print(f"文件包含 {len(embeddings)} 条记录，每条记录的向量维度为 {embeddings.shape[1]}")

def save_embeddings_npy(output_dir, output_prefix, embeddings):
    """
    将嵌入向量保存为.npy文件
    
    Args:
        output_dir: 输出目录
        output_prefix: 输出文件前缀
        embeddings: 嵌入向量数组
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为npy文件
    output_path = os.path.join(output_dir, f"{output_prefix}.npy")
    np.save(output_path, embeddings)
    
    print(f"嵌入向量已保存为.npy文件: {output_path}")
    print(f"文件包含 {len(embeddings)} 条记录，每条记录的向量维度为 {embeddings.shape[1]}")


if __name__ == "__main__":
    
    #     # 脚本文件所在目录
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # # raw.csv 的绝对路径
    # csv_path = os.path.join(BASE_DIR, "raw.csv")
    
    # # 读取csv文件
    # df = pd.read_csv(csv_path)
    
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 设置参数
    input_file = os.path.join(script_dir, "raw.csv")  # 使用绝对路径
    output_dir = script_dir  # 直接输出到脚本所在目录
    
    print(f"输入文件路径: {input_file}")
    print(f"输出目录路径: {output_dir}")
    
    # 文本嵌入参数
    text_model_name = "NeuML/pubmedbert-base-embeddings"  # 文本嵌入模型
    batch_size = 256                         # 批处理大小
    
    # MeSH术语嵌入参数
    mesh_model_name = "dmis-lab/biobert-v1.1"  # MeSH术语嵌入模型
    
    # 加载数据（只需要加载一次）
    df, pmids, texts, mesh_terms_list = load_cleaned_data(input_file)
    
    # 嵌入文本（只需要嵌入一次）
    text_embeddings = embed_texts(texts, model_name=text_model_name, batch_size=batch_size)
    
    # 定义alpha和beta的值范围
    # alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # beta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # 添加更细粒度的值作为补充
    alpha_values = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    beta_values = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    # 生成所有组合
    combinations = list(itertools.product(alpha_values, beta_values))
    
    print(f"将生成 {len(combinations)} 个不同的alpha和beta组合嵌入向量")
    
    # 对每个beta值嵌入一次MeSH术语
    mesh_embeddings_cache = {}
    for beta in beta_values:
        print(f"使用beta={beta}嵌入MeSH术语...")
        mesh_embeddings, has_mesh = embed_mesh_terms(
            mesh_terms_list, 
            beta=beta,
            model_name=mesh_model_name, 
            batch_size=batch_size
        )
        mesh_embeddings_cache[beta] = (mesh_embeddings, has_mesh)
    
    # 存储生成的所有文件路径
    generated_files = []
    
    # 对每个组合生成融合嵌入
    for alpha, beta in combinations:
        print(f"\n处理组合: alpha={alpha}, beta={beta}")
        
        # 从缓存中获取对应beta值的MeSH嵌入
        mesh_embeddings, has_mesh = mesh_embeddings_cache[beta]
        
        # 合并嵌入
        combined_embeddings = combine_embeddings(
            text_embeddings, 
            mesh_embeddings, 
            has_mesh,
            alpha=alpha
        )
        
        # 根据alpha和beta的值生成输出文件前缀
        alpha_str = str(int(alpha * 100))
        beta_str = str(int(beta * 100))
        output_prefix = f"emb_a{alpha_str}b{beta_str}"  # 例如：emb_a75b75
        
        # 保存嵌入向量为.npy文件
        save_embeddings_npy(output_dir, output_prefix, combined_embeddings)
        
        # 将文件路径添加到列表
        file_path = os.path.join(output_dir, f"{output_prefix}.npy")
        generated_files.append(file_path)
    
    print("所有组合的嵌入向量处理完成！")
    print("\n生成的文件列表:")
    for file_path in generated_files:
        print(file_path) 