import os
import subprocess
import venv
from config.log_config import logger

try:
    # 1. Create virtual environment with pip
    venv_dir = "venv"
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(venv_dir)

    # 2. Specify path to python and pip in venv
    if os.name == 'nt':  # Windows
        python_exe = os.path.join(venv_dir, 'Scripts', 'python.exe')
        pip_exe = os.path.join(venv_dir, 'Scripts', 'pip.exe')
    else:
        python_exe = os.path.join(venv_dir, 'bin', 'python')
        pip_exe = os.path.join(venv_dir, 'bin', 'pip')

    # 3. Check if the system supports CUDA (NVIDIA GPU)
    cuda_supported = (subprocess.run(
        ['nvidia-smi'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0)

    # 4. Install PyTorch
    if subprocess.run([pip_exe, 'show', 'torch'], stdout=subprocess.DEVNULL).returncode != 0:
        if cuda_supported:
            subprocess.run([pip_exe, 'install', 'torch', '--index-url', 'https://download.pytorch.org/whl/cu128'])
        else:
            subprocess.run([pip_exe, 'install', 'torch'])

    # 5. Install the necessary libraries
    subprocess.run([pip_exe, 'install', '-r', 'requirements.txt'])
    logger.info("Cài đặt thành công!")
except Exception as e:
    logger.error(f"Có lỗi xảy ra trong quá trình cài đặt:\n{e}")
