import requests
from config.log_config import logger
from config.load_env import env
import os
from tqdm import tqdm
import requests.utils
from bs4 import BeautifulSoup

local_dir = os.path.join(os.getcwd(), 'data')
os.makedirs(local_dir, exist_ok=True)

try:
    logger.info(f"{os.path.basename(__file__)}: Bắt đầu quá trình tải tệp từ ID: {env.FILE_ID}")
    initial_url = f'https://drive.google.com/uc?export=download&id={env.FILE_ID}'
    response = requests.get(initial_url, stream=True, allow_redirects=True)
    response.raise_for_status()
    content_type = response.headers.get('Content-Type', '').lower()

    if 'text/html' in content_type:
        logger.info(f"{os.path.basename(__file__)}: Nhận được phản hồi là HTML.")
        html_content = response.content.decode('utf-8', errors='ignore')
        response.close()
        response = None 
        soup = BeautifulSoup(html_content, 'html.parser')
        download_form = soup.find('form', {'id': 'download-form'})
        if download_form:
            form_action = download_form.get('action')
            if not form_action:
                 raise Exception("Không tìm thấy action URL trong form xác nhận.")
            hidden_inputs = download_form.find_all('input', {'type': 'hidden'})
            form_params = {}
            for input_tag in hidden_inputs:
                input_name = input_tag.get('name')
                input_value = input_tag.get('value')
                if input_name and input_value is not None:
                    form_params[input_name] = input_value
            final_download_url = form_action
            response = requests.get(final_download_url, params=form_params, stream=True, allow_redirects=True)
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type', '').lower():
                logger.error(f"{os.path.basename(__file__)}: Phản hồi lần 2 vẫn là HTML. Có thể có lỗi hoặc trang xác nhận khác.")
                logger.error(f"{os.path.basename(__file__)}: Headers: {response.headers}")
                raise Exception("Phản hồi lần 2 vẫn là HTML.")
        else:
            logger.error(f"{os.path.basename(__file__)}: Nhận được HTML nhưng không tìm thấy form tải xuống mong đợi (id='download-form').")
            raise Exception("Không tìm thấy form tải xuống trong trang xác nhận.")
    logger.info(f"{os.path.basename(__file__)}: Đã sẵn sàng tải tệp. Đang kiểm tra header tệp...")
    if response.headers.get('Content-Length', None):
        try:
            total_size = int(response.headers.get('Content-Length'))
            logger.info(f"{os.path.basename(__file__)}: Kích thước tệp: {total_size} bytes.")
        except ValueError:
             total_size = None
    else:
        total_size = None 
    content_disposition = response.headers.get('Content-Disposition')
    if content_disposition:
        try:
            parts = content_disposition.split(';')
            for part in parts:
                part = part.strip()
                if part.lower().startswith('filename='):
                    filename = requests.utils.unquote(part.split('=', 1)[1].strip().strip('"'))
                    break
                elif part.lower().startswith('filename*='):
                    try:
                        encoded_part = part.split('=', 1)[1]
                        if "''" in encoded_part:
                            encoding, _, encoded_filename = encoded_part.split("''", 1)
                            filename = requests.utils.unquote(encoded_filename)
                        else:
                            filename = requests.utils.unquote(encoded_part)
                    except Exception as e:
                        logger.warning(f"{os.path.basename(__file__)}: Lỗi khi phân tích filename* trong Content-Disposition '{part}': {e}")
                        break
            if not filename and 'filename=' in content_disposition:
                try:
                    filename_match = requests.utils.re.search(r'filename\*?=(?:"([^"]*)"|([^;]+))', content_disposition)
                    if filename_match:
                        filename = requests.utils.unquote(filename_match.group(1) or filename_match.group(2)).strip()
                        logger.debug(f"{os.path.basename(__file__)}: Fallback filename parsing successful: {filename}")
                except Exception as e:
                    logger.warning(f"{os.path.basename(__file__)}: Lỗi khi thử fallback parsing filename: {e}")
            if filename is None or not filename.strip():
                logger.warning(f"{os.path.basename(__file__)}: Không thể tìm thấy filename hợp lệ trong Content-Disposition header. Sử dụng tên mặc định.")
                filename = None 
        except Exception as e:
            logger.warning(f"{os.path.basename(__file__)}: Lỗi tổng quát khi phân tích header Content-Disposition '{content_disposition}': {e}. Sử dụng tên mặc định.")
            filename = None 
    if not filename:
        filename = f'downloaded_file_{env.FILE_ID}'
        logger.warning(f"{os.path.basename(__file__)}: Sử dụng tên tệp mặc định: {filename}")
    else:
        filename = os.path.basename(filename)
        logger.debug(f"{os.path.basename(__file__)}: Tên tệp sau khi vệ sinh: {filename}")
    local_file_path = os.path.join(local_dir, filename)
    logger.info(f"{os.path.basename(__file__)}: Bắt đầu tải xuống...")
    if response is None:
         raise Exception("Không nhận được phản hồi hợp lệ để tải xuống tệp.")
    with open(local_file_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Tải xuống [{filename}]", leave=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: 
                    f.write(chunk)
                    pbar.update(len(chunk)) 
    logger.info(f"{os.path.basename(__file__)}: Đã tải tệp và lưu tại: {local_file_path}")

except requests.exceptions.RequestException as e:
    logger.error(f"{os.path.basename(__file__)}: Lỗi khi tải tệp công khai: {e}")
    if response is not None:
        logger.error(f"{os.path.basename(__file__)}: URL yêu cầu lỗi: {response.request.url}")
        logger.error(f"{os.path.basename(__file__)}: Status Code: {response.status_code}")
        logger.error(f"{os.path.basename(__file__)}: Headers: {response.headers}")
        if response.status_code == 403:
             logger.error(f"{os.path.basename(__file__)}: Lỗi 403 Forbidden. Có thể tệp không được chia sẻ công khai ('Anyone with the link can view') hoặc có giới hạn tải xuống.")
        elif response.status_code == 404:
             logger.error(f"{os.path.basename(__file__)}: Lỗi 404 Not Found. Kiểm tra lại File ID.")
        elif response.status_code == 429:
             logger.error(f"{os.path.basename(__file__)}: Lỗi 429 Too Many Requests. Đã vượt quá giới hạn tải xuống cho tệp này.")
        else:
             logger.error(f"{os.path.basename(__file__)}: Lỗi HTTP không xác định: {response.status_code}.")
    else:
        logger.error(f"{os.path.basename(__file__)}: Lỗi Requests xảy ra trước khi nhận được phản hồi.")
except Exception as e:
    logger.error(f"{os.path.basename(__file__)}: Đã xảy ra lỗi không mong muốn: {e}")
    import traceback
    logger.error(f"{os.path.basename(__file__)}: Traceback: {traceback.format_exc()}") 
finally:
    if response:
        response.close()
        logger.debug(f"{os.path.basename(__file__)}: Đã hoàn tất quá trình tải tệp.")