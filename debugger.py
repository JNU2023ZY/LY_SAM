import subprocess
import os

# 初始化 VOTS2023
init_command = 'vot initialize vots2023 --workspace /data_F/zhouyong/DMAOT/dmaot'
try:
    subprocess.run(init_command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error executing initialization command: {e}")
    exit(1)

# 执行 evaluate.sh 并获取输出
sh_file = 'evaluate.sh'
sh_file_path = '/data_F/zhouyong/DMAOT/DMAOT-VOTS2023-master/' + sh_file

# 检查 sh 文件是否存在
if not os.path.exists(sh_file_path):
    print(f"Error: File '{sh_file_path}' not found.")
    exit(1)

try:
    output = subprocess.check_output(['bash', sh_file_path])

    # 创建新的 Python 文件并写入命令输出结果
    py_file = 'evaluate.py'
    with open(py_file, 'w') as f:
        f.write(output.decode('utf-8'))

    # 在新的 Python 文件中继续编写
    with open(py_file, 'a') as f:
        f.write('\n\nimport subprocess\n\n')
        f.write(f'subprocess.run(["bash", "{sh_file_path}"])')

    # 运行新的 Python 文件
    os.system('python {}'.format(py_file))

except subprocess.CalledProcessError as e:
    print(f"Error executing bash script: {e}")