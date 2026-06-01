import os

def generate_llm_context(root_dir=".", output_filename="llm_context.xml"):
    # 1. 定义需要严格剔除的目录（精确匹配）
    exclude_dirs = {
        '.git', 'node_modules', 'venv', 'env', '__pycache__', 
        '.vscode', '.idea', 'dataset', 'datasets', 'test', 'tests', 
        'logs', 'backup', 'build', 'dist'
    }
    
    # 2. 定义需要跳过的非代码扩展名（防止将权重、日志、压缩包读入内存）
    exclude_exts = {
        # 模型与数据
        '.pth', '.pt', '.onnx', '.pb', '.npy', '.npz', '.csv', '.jsonl',
        # 媒体与压缩包
        '.mp4', '.avi', '.jpg', '.png', '.zip', '.tar', '.gz',
        # 编译与二进制文件
        '.pyc', '.so', '.dll', '.exe', '.bin'
    }

    total_files_processed = 0
    
    print(f"开始扫描项目: {os.path.abspath(root_dir)}")
    
    with open(output_filename, 'w', encoding='utf-8') as out_file:
        # 写入根标签
        out_file.write("<project_context>\n")

        for dirpath, dirnames, filenames in os.walk(root_dir):
            # 原地修改 dirnames，这样 os.walk 就不会去遍历被排除的文件夹，极大提升扫描速度
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                
                # 排除指定扩展名的文件
                if ext in exclude_exts:
                    continue

                file_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(file_path, root_dir)
                
                # 排除脚本自身和生成的输出文件
                if filename == "pack_context.py" or filename == output_filename:
                    continue

                try:
                    # 尝试以 UTF-8 读取文件
                    with open(file_path, 'r', encoding='utf-8') as in_file:
                        content = in_file.read()

                    # 写入 XML 结构
                    out_file.write(f'  <file path="{rel_path}">\n')
                    out_file.write(content)
                    # 确保文件内容以换行符结束，避免标签错乱
                    if not content.endswith('\n'):
                        out_file.write('\n')
                    out_file.write(f'  </file>\n\n')
                    
                    total_files_processed += 1
                    
                except UnicodeDecodeError:
                    # 如果不是纯文本文件（比如一些没有后缀的二进制文件），静默跳过
                    pass
                except Exception as e:
                    print(f"[警告] 读取文件失败 {rel_path}: {e}")

        # 闭合根标签
        out_file.write("</project_context>\n")

    print(f"✅ 打包完成！共处理了 {total_files_processed} 个文件。")
    print(f"📁 输出文件位置: {os.path.abspath(output_filename)}")
    print(f"📊 提示：可以直接将 {output_filename} 拖入或复制给大模型。")

if __name__ == "__main__":
    # 执行脚本，默认扫描当前目录，输出为 llm_context.xml
    generate_llm_context()