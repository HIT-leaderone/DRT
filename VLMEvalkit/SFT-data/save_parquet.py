import json
import os
import pandas as pd
import shutil
import subprocess
from tqdm import tqdm

INPUT_JSON_PATH = os.environ.get("VISION_R1_INPUT_JSON", "./Video-R1-COT-165k.json")
VIDEO_BASE_PATH = os.environ.get("VISION_R1_MEDIA_ROOT", "./Video-R1-data")
OUTPUT_DIR = os.environ.get("VISION_R1_OUTPUT_DIR", "./Video-R1-parquet")
OUTPUT_FILE_BASE_NAME = os.environ.get("VISION_R1_OUTPUT_BASENAME", "Video-R1")

# [修改点] 分别定义分块大小
VIDEO_CHUNK_SIZE = 1000   # 视频较大，保持 1000
IMAGE_CHUNK_SIZE = 10000  # 图片较小，增加到 10000

# 本地临时目录
LOCAL_TEMP_DIR = os.environ.get("VISION_R1_TEMP_DIR", "./temp_parquet_buffer")

# 支持的后缀名定义
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# 定义 Prompt 模板
DENSE_PROMPT_TEMPLATE = (
    "{Question}\n"
    "{Options}\n"
    "Analyze this question with **maximum information density** and strict logical precision.\n"
    "1. **Format**: Use a **telegraphic style** (concise phrases, arrows '->', symbols).\n"
    "2. **Constraint**: STRICTLY FORBIDDEN to use conversational fillers (e.g., 'Let me think', 'Hmm', 'I see', 'Wait').\n"
    "3. **Content**: Focus only on: [Key Visual Evidence] -> [Logical Inference] -> [Conclusion].\n"
    "Provide your dense cognitive trace between the <think> </think> tags, and then give your final answer."
)

# DENSE_PROMPT_TEMPLATE = (
#     "{Question}\n"
#     "Please think about this question as if you were a human pondering deeply. "
#     "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
#     "It's encouraged to include self-reflection or verification in the reasoning process. "
#     "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
# )

def format_options(options_list):
    if not options_list:
        return ""
    return "\n".join(str(opt) for opt in options_list)

def init_dirs():
    """初始化本地临时目录和输出目录"""
    if not os.path.exists(LOCAL_TEMP_DIR):
        os.makedirs(LOCAL_TEMP_DIR)

    if OUTPUT_DIR.startswith("hdfs://"):
        print("[*] Initializing HDFS directories...")
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", f"{OUTPUT_DIR}/video"], check=True)
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", f"{OUTPUT_DIR}/image"], check=True)
        print("[*] HDFS directories ready.")
    else:
        os.makedirs(os.path.join(OUTPUT_DIR, "video"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "image"), exist_ok=True)

def save_chunk(item_list, data_type, output_dir, base_name, chunk_index):
    """
    保存分块数据：先存本地 -> Shell上传 -> 删本地
    """
    if not item_list:
        return

    df = pd.DataFrame(item_list)

    # 文件名
    file_name = f"{base_name}_part_{chunk_index}.parquet"

    # 1. 本地路径
    local_path = os.path.join(LOCAL_TEMP_DIR, file_name)

    dest_path = f"{output_dir}/{data_type}/{file_name}"

    try:
        # A. 保存到本地
        df.to_parquet(local_path, engine='pyarrow', index=False)

        if output_dir.startswith("hdfs://"):
            ret = subprocess.run(
                ["hdfs", "dfs", "-put", "-f", local_path, dest_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if ret.returncode != 0:
                print(f"   [ERROR] HDFS upload failed for {file_name}. Error: {ret.stderr.decode()}")
                return
            os.remove(local_path)
        else:
            shutil.move(local_path, dest_path)
        print(f"   [Saved] {data_type} chunk {chunk_index} ({len(item_list)} rows) -> {dest_path}")

    except Exception as e:
        print(f"   [ERROR] Failed to process chunk {file_name}: {e}")

def process_data_to_parquet():
    # 0. 初始化目录
    init_dirs()

    # 1. 读取 JSON 数据
    print(f"[*] Loading JSON from: {INPUT_JSON_PATH}")
    if not os.path.exists(INPUT_JSON_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_JSON_PATH}")

    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"[*] Total records: {len(data)}")

    # 缓冲区和计数器
    video_buffer = []
    image_buffer = []

    video_chunk_idx = 0
    image_chunk_idx = 0

    missing_files = 0

    # 2. 遍历处理
    for item in tqdm(data, desc="Processing Records"):
        # --- A. 获取路径与判断类型 ---
        rel_path = item.get('path', '').lstrip('./')
        full_media_path = os.path.join(VIDEO_BASE_PATH, rel_path)

        _, ext = os.path.splitext(full_media_path)
        ext = ext.lower()

        media_type = None
        media_token = ""

        if ext in VIDEO_EXTENSIONS:
            media_type = 'video'
            media_token = "<video>"
            continue
        elif ext in IMAGE_EXTENSIONS:
            media_type = 'image'
            media_token = "<image>"
        else:
            continue
        # --- B. 读取二进制数据 ---
        media_bytes = None
        if os.path.exists(full_media_path):
            try:
                with open(full_media_path, 'rb') as mf:
                    media_bytes = mf.read()
            except Exception as e:
                print(f"[!] Error reading {full_media_path}: {e}")
                continue
        else:
            missing_files += 1
            continue

        if media_bytes is None:
            continue

        # --- C. 构建 Conversation ---
        question_text = item.get('problem', '')
        options_text = format_options(item.get('options', []))

        user_text = DENSE_PROMPT_TEMPLATE.format(
            Question=question_text,
            Options=options_text
        )

        final_user_content = media_token + "\n" + user_text

        raw_process = item.get('process', '').strip()
        if not raw_process.startswith("<think>"):
            raw_process = "<think>\n" + raw_process
        if not raw_process.endswith("</think>"):
            raw_process = raw_process + "\n</think>"

        raw_solution = item.get('solution', '')
        clean_solution = raw_solution.replace("<answer>", "").replace("</answer>", "").strip()

        final_assistant_content = f"{raw_process}\n{clean_solution}"

        conversation = [
            {"role": "user", "content": final_user_content},
            {"role": "assistant", "content": final_assistant_content}
        ]

        row_item = {
            "problem_id": item.get('problem_id'),
            "conversation": conversation,
            media_type: media_bytes
        }

        # --- D. 存入缓冲区并检查是否需要保存 ---
        if media_type == 'video':
            video_buffer.append(row_item)
            # [修改点] 使用 VIDEO_CHUNK_SIZE (1000)
            if len(video_buffer) >= VIDEO_CHUNK_SIZE:
                save_chunk(video_buffer, 'video', OUTPUT_DIR, OUTPUT_FILE_BASE_NAME, video_chunk_idx)
                video_buffer = []
                video_chunk_idx += 1
        else:
            image_buffer.append(row_item)
            # [修改点] 使用 IMAGE_CHUNK_SIZE (10000)
            if len(image_buffer) >= IMAGE_CHUNK_SIZE:
                save_chunk(image_buffer, 'image', OUTPUT_DIR, OUTPUT_FILE_BASE_NAME, image_chunk_idx)
                image_buffer = []
                image_chunk_idx += 1

    # 3. 循环结束后，保存剩余的数据 (Flush leftovers)
    if video_buffer:
        save_chunk(video_buffer, 'video', OUTPUT_DIR, OUTPUT_FILE_BASE_NAME, video_chunk_idx)

    if image_buffer:
        save_chunk(image_buffer, 'image', OUTPUT_DIR, OUTPUT_FILE_BASE_NAME, image_chunk_idx)

    # 4. 清理本地临时目录
    try:
        os.rmdir(LOCAL_TEMP_DIR)
    except:
        pass

    print(f"\n[*] Processing complete.")
    print(f"    - Missing Files: {missing_files}")
    print(f"    - Total Video Chunks: {video_chunk_idx + (1 if video_buffer else 0)}")
    print(f"    - Total Image Chunks: {image_chunk_idx + (1 if image_buffer else 0)}")

if __name__ == "__main__":
    process_data_to_parquet()
