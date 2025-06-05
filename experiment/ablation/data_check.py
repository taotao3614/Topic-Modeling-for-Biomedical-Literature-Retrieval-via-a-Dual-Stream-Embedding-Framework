import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# 设置输入文件路径（使用简单的路径）
input_file = "ablation/raw.csv"  # 直接指向ablation文件夹下的raw.csv

def check_data():
    """
    检查depression.csv数据完整性
    1. 检查每列数据的缺失情况
    2. 检查年份数据是否存在小数点问题
    3. 检查mesh_terms是否为空
    4. 检测数值异常并输出对应PMID
    5. 修复年份数据（转换为整型）并保存回原文件
    """
    file_path = input_file
    print(f"开始读取数据文件: {file_path}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(file_path)
        print(f"成功读取数据，共有 {len(df)} 行数据")
    except Exception as e:
        print(f"读取数据文件出错: {e}")
        return
    
    # 输出数据检查报告到控制台
    print("\n" + "="*50)
    print(f"数据检查报告 - 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据文件: {file_path}")
    print(f"总行数: {len(df)}")
    print("="*50 + "\n")
    
    # 检查PMID列是否存在
    has_pmid = 'pmid' in df.columns
    if not has_pmid:
        print("警告: 数据集中没有'pmid'列，无法输出异常值对应的PMID\n")
    
    # 1. 检查每列数据的缺失情况
    print("1. 数据完整性检查")
    print("-" * 50)
    
    for column in df.columns:
        missing = df[column].isna().sum()
        missing_percentage = (missing / len(df)) * 100
        print(f"列名: {column}")
        print(f"  - 数据类型: {df[column].dtype}")
        print(f"  - 缺失值数量: {missing} ({missing_percentage:.2f}%)")
        print(f"  - 非空值数量: {len(df) - missing} ({100 - missing_percentage:.2f}%)")
        
        # 显示一些基本统计信息（如果适用）
        if pd.api.types.is_numeric_dtype(df[column]):
            print(f"  - 最小值: {df[column].min()}")
            print(f"  - 最大值: {df[column].max()}")
            print(f"  - 平均值: {df[column].mean()}")
            print(f"  - 中位数: {df[column].median()}")
            
            # 检测并输出数值异常
            if has_pmid and column != 'pmid':
                # 使用IQR方法检测异常值
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                if len(outliers) > 0:
                    print(f"  - 检测到 {len(outliers)} 个异常值 (超出范围: {lower_bound:.2f} - {upper_bound:.2f})")
                    print("  - 异常值对应的PMID:")
                    for idx, row in outliers.head(10).iterrows():
                        print(f"     PMID: {row['pmid']}, 值: {row[column]}")
                    if len(outliers) > 10:
                        print(f"     ... 还有 {len(outliers) - 10} 个异常值未显示")
        elif column == 'mesh_terms':
            empty_mesh = df[df[column].notna()].apply(
                lambda x: len(x[column]) <= 2 if isinstance(x[column], str) else True, 
                axis=1
            ).sum()
            print(f"  - 空的mesh_terms数量: {empty_mesh} ({(empty_mesh / len(df)) * 100:.2f}%)")
            
            # 如果存在PMID列，输出空mesh_terms对应的PMID
            if has_pmid:
                empty_mesh_df = df[df[column].apply(
                    lambda x: pd.isna(x) or (isinstance(x, str) and len(x) <= 2)
                )]
                if len(empty_mesh_df) > 0:
                    print("  - 空mesh_terms对应的PMID样例:")
                    for idx, row in empty_mesh_df.head(10).iterrows():
                        print(f"     PMID: {row['pmid']}")
                    if len(empty_mesh_df) > 10:
                        print(f"     ... 还有 {len(empty_mesh_df) - 10} 条记录未显示")
        
        print()
    
    # 2. 检查年份数据中的小数点问题
    if 'pub_year' in df.columns:
        print("\n2. 年份数据检查")
        print("-" * 50)
        print(f"年份数据统计: \n{df['pub_year'].describe()}")
        
        # 检查是否存在小数点
        has_decimal = (df['pub_year'] % 1 != 0).any()
        if has_decimal:
            decimal_rows = df[df['pub_year'] % 1 != 0]
            decimal_count = len(decimal_rows)
            print(f"注意: 发现年份数据中存在小数点")
            print(f"含小数点的年份数量: {decimal_count} ({(decimal_count / len(df)) * 100:.2f}%)")
            
            # 如果存在PMID列，输出含小数点年份对应的PMID
            if has_pmid:
                print("含小数点年份对应的PMID:")
                for idx, row in decimal_rows.head(10).iterrows():
                    print(f"  PMID: {row['pmid']}, 年份值: {row['pub_year']}")
                if decimal_count > 10:
                    print(f"  ... 还有 {decimal_count - 10} 条记录未显示")
                    
        # 检查异常年份（比当前年份大的或过小的年份）
        current_year = datetime.now().year
        future_years = df[df['pub_year'] > current_year]
        if len(future_years) > 0:
            print(f"\n发现 {len(future_years)} 个超过当前年份({current_year})的记录")
            if has_pmid:
                print("对应的PMID:")
                for idx, row in future_years.head(10).iterrows():
                    print(f"  PMID: {row['pmid']}, 年份值: {row['pub_year']}")
                if len(future_years) > 10:
                    print(f"  ... 还有 {len(future_years) - 10} 条记录未显示")
        
        too_old_years = df[df['pub_year'] < 1900]
        if len(too_old_years) > 0:
            print(f"\n发现 {len(too_old_years)} 个年份小于1900的记录")
            if has_pmid:
                print("对应的PMID:")
                for idx, row in too_old_years.head(10).iterrows():
                    print(f"  PMID: {row['pmid']}, 年份值: {row['pub_year']}")
                if len(too_old_years) > 10:
                    print(f"  ... 还有 {len(too_old_years) - 10} 条记录未显示")
    
    # 3. 检查mesh_terms是否为空
    if 'mesh_terms' in df.columns:
        print("\n3. Mesh Terms检查")
        print("-" * 50)
        
        # 计算空或无效的mesh_terms (空字符串、"[]"、null等)
        empty_na = df['mesh_terms'].isna().sum()
        empty_brackets = (df['mesh_terms'] == '[]').sum()
        empty_string = (df['mesh_terms'] == '').sum()
        
        print(f"空值(NA)数量: {empty_na} ({(empty_na / len(df)) * 100:.2f}%)")
        print(f"空括号([])数量: {empty_brackets} ({(empty_brackets / len(df)) * 100:.2f}%)")
        print(f"空字符串数量: {empty_string} ({(empty_string / len(df)) * 100:.2f}%)")
        
        # 如果存在PMID列，输出各类空mesh_terms对应的PMID示例
        if has_pmid:
            if empty_na > 0:
                na_pmids = df[df['mesh_terms'].isna()]['pmid'].head(5).tolist()
                print(f"空值(NA)对应的PMID示例: {na_pmids}")
            
            if empty_brackets > 0:
                bracket_pmids = df[df['mesh_terms'] == '[]']['pmid'].head(5).tolist()
                print(f"空括号([])对应的PMID示例: {bracket_pmids}")
            
            if empty_string > 0:
                string_pmids = df[df['mesh_terms'] == '']['pmid'].head(5).tolist()
                print(f"空字符串对应的PMID示例: {string_pmids}")
        
        # 检查空JSON数组
        def is_empty_json_array(mesh_term):
            if not isinstance(mesh_term, str):
                return False
            try:
                parsed = json.loads(mesh_term)
                return len(parsed) == 0
            except:
                return False
        
        empty_json = df['mesh_terms'].apply(is_empty_json_array).sum()
        print(f"空JSON数组数量: {empty_json} ({(empty_json / len(df)) * 100:.2f}%)")
        
        if has_pmid and empty_json > 0:
            json_pmids = df[df['mesh_terms'].apply(is_empty_json_array)]['pmid'].head(5).tolist()
            print(f"空JSON数组对应的PMID示例: {json_pmids}")
        
        total_empty = empty_na + empty_brackets + empty_string + empty_json
        print(f"总空mesh_terms数量: {total_empty} ({(total_empty / len(df)) * 100:.2f}%)")
    
    # 4. 检查title和abstract是否异常短
    if 'title' in df.columns:
        print("\n4. 标题长度检查")
        print("-" * 50)
        
        # 计算标题长度
        df['title_length'] = df['title'].fillna('').apply(len)
        avg_len = df['title_length'].mean()
        min_len = df['title_length'].min()
        
        print(f"标题平均长度: {avg_len:.2f} 字符")
        print(f"标题最短长度: {min_len} 字符")
        
        # 检测异常短的标题
        very_short_titles = df[df['title_length'] < 10]
        if len(very_short_titles) > 0:
            print(f"发现 {len(very_short_titles)} 条标题长度小于10个字符的记录")
            if has_pmid:
                print("对应的PMID和标题:")
                for idx, row in very_short_titles.head(10).iterrows():
                    print(f"  PMID: {row['pmid']}, 标题: \"{row['title']}\"")
                if len(very_short_titles) > 10:
                    print(f"  ... 还有 {len(very_short_titles) - 10} 条记录未显示")
        
        # 删除临时列
        df.drop('title_length', axis=1, inplace=True)
    
    if 'abstract' in df.columns:
        print("\n5. 摘要长度检查")
        print("-" * 50)
        
        # 计算摘要长度
        df['abstract_length'] = df['abstract'].fillna('').apply(len)
        avg_len = df['abstract_length'].mean()
        min_len = df['abstract_length'].min()
        
        print(f"摘要平均长度: {avg_len:.2f} 字符")
        print(f"摘要最短长度: {min_len} 字符")
        
        # 检测空的摘要
        empty_abstract = df[df['abstract_length'] == 0]
        if len(empty_abstract) > 0:
            print(f"发现 {len(empty_abstract)} 条空摘要记录 ({(len(empty_abstract) / len(df)) * 100:.2f}%)")
            if has_pmid:
                print("对应的PMID示例:")
                for idx, row in empty_abstract.head(10).iterrows():
                    print(f"  PMID: {row['pmid']}")
                if len(empty_abstract) > 10:
                    print(f"  ... 还有 {len(empty_abstract) - 10} 条记录未显示")
        
        # 检测异常短的摘要
        very_short_abstract = df[(df['abstract_length'] > 0) & (df['abstract_length'] < 50)]
        if len(very_short_abstract) > 0:
            print(f"发现 {len(very_short_abstract)} 条摘要长度小于50个字符的记录")
            if has_pmid:
                print("对应的PMID和摘要:")
                for idx, row in very_short_abstract.head(5).iterrows():
                    print(f"  PMID: {row['pmid']}, 摘要: \"{row['abstract']}\"")
                if len(very_short_abstract) > 5:
                    print(f"  ... 还有 {len(very_short_abstract) - 5} 条记录未显示")
        
        # 删除临时列
        df.drop('abstract_length', axis=1, inplace=True)
    
    print("\n数据检查完成")
    
    # 5. 修复年份数据（如果存在pub_year列）
    if 'pub_year' in df.columns:
        print("\n开始修复年份数据...")
        
        # 备份原始文件
        backup_file = file_path + ".bak"
        df.to_csv(backup_file, index=False)
        print(f"已备份原始文件到: {backup_file}")
        
        # 转换年份数据为整数
        original_values = df['pub_year'].copy()
        df['pub_year'] = df['pub_year'].fillna(0).astype(int)
        
        # 检查修复前后的差异
        changed = (original_values != df['pub_year']).sum()
        print(f"已将 {changed} 个年份记录从浮点数转换为整数")
        
        # 保存修改后的数据
        df.to_csv(file_path, index=False)
        print(f"已保存修复后的数据到原始文件: {file_path}")
        
        # 再次检查是否有浮点数
        df_check = pd.read_csv(file_path)
        if (df_check['pub_year'] % 1 != 0).any():
            print("警告: 保存后的文件仍有浮点数年份，请检查数据类型")
        else:
            print("成功: 所有年份数据已成功转换为整数")
    
    # 6. 添加text字段（title和abstract的拼接）
    if 'title' in df.columns and 'abstract' in df.columns:
        print("\n6. 添加text字段（title和abstract的拼接）")
        print("-" * 50)
        
        # 使用fillna确保没有缺失值，然后拼接title和abstract
        df['text'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
        
        # 清理text字段中的多余空格
        df['text'] = df['text'].str.strip()
        
        # 计算text字段的长度统计
        text_lengths = df['text'].apply(len)
        avg_len = text_lengths.mean()
        min_len = text_lengths.min()
        max_len = text_lengths.max()
        
        print(f"已成功添加text字段（title和abstract的拼接）")
        print(f"text字段长度统计:")
        print(f"  - 平均长度: {avg_len:.2f} 字符")
        print(f"  - 最短长度: {min_len} 字符")
        print(f"  - 最长长度: {max_len} 字符")
        
        # 保存更新后的数据
        df.to_csv(file_path, index=False)
        print(f"已保存添加text字段后的数据到原始文件: {file_path}")
        
        # 输出前5条记录的text示例
        print("\ntext字段示例（前5条记录）:")
        for i, text in enumerate(df['text'].head(5)):
            print(f"记录 {i+1}: {text[:100]}{'...' if len(text) > 100 else ''}")
    else:
        print("\n警告: 数据集中没有title或abstract列，无法添加text字段")

if __name__ == "__main__":
    check_data() 