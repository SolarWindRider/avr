# ../datas/VisuRiddles目录下运行这个脚本
# 用于数据处理，并分割训练集和测试集
import json
import random
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw, ImageFont
import os


def raven_cat(
    img_id,
    input_dir=".",
    output_dir=".",
    border_width=2,
    border_color=(0, 0, 0),
    label_color=(0, 0, 0),
    label_font_size=40,
    gap_width=20,
):
    """
    简化的图像拼接函数：将question image和ground truth image按照长边拼接，并添加文字标识。

    Parameters:
    - image_id: 图像ID (e.g., '768')
    - input_dir: 输入图像目录 (default: 当前目录)
    - output_dir: 输出图像目录 (default: 当前目录)
    - border_width: 边框宽度
    - border_color: 边框颜色RGB元组
    - label_color: 标签颜色RGB元组
    - label_font_size: 标签字体大小
    - gap_width: 两个图像之间的间距
    """

    def add_text_label(img, text, font_size=40, position="top"):
        """在图像上添加文字标签"""
        draw = ImageDraw.Draw(img)
        font = None
        # 尝试加载常见系统字体路径（DejaVu 是 Linux/Ubuntu/ModelArts 中常见的字体）
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # 常见 Ubuntu 路径
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",   # 另一个常见备选
        ]
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, font_size)
                print(f"成功加载字体: {path}")
                break
            except IOError:
                pass  # 继续尝试下一个

        # 如果系统路径失败，尝试使用 matplotlib 来查找字体路径（如果 matplotlib 已安装）
        if font is None:
            try:
                import matplotlib.font_manager as fm
                font_path = fm.findfont('DejaVu Sans')  # 查找 DejaVu 或类似 sans 字体
                font = ImageFont.truetype(font_path, font_size)
                print(f"成功通过 matplotlib 加载字体: {font_path}")
            except (ImportError, IOError):
                pass

        # 最终回退到默认字体，但警告它不会缩放
        if font is None:
            font = ImageFont.load_default()
            print("警告: 使用默认字体，无法缩放大小。建议安装 DejaVu 字体或 matplotlib。")

        # 计算文字位置
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if position == "top":
            x = (img.width - text_w) // 2
            y = 5  # 减少顶部边距，让标签更贴近图像顶部
        else:  # left
            x = 10
            y = (img.height - text_h) // 2

        # 添加白色背景（无边框）
        padding = font_size // 8  # 根据字体大小动态调整填充（更大字体需要更多空间）
        draw.rectangle(
            [x - padding, y - padding, x + text_w + padding, y + text_h + padding], fill=(255, 255, 255)
        )
        draw.text((x, y), text, font=font, fill=label_color)
        return img

    # 定义文件路径
    input_path = os.path.join(input_dir, f"{img_id}.jpg")
    gt_path = os.path.join(input_dir, f"{img_id}_gt.jpg")
    output_path = output_dir + "/" + f"{img_id}_concat.jpg"

    # 加载图像
    question_img = Image.open(input_path)
    gt_img = Image.open(gt_path)

    # 确保图像是RGB模式
    if question_img.mode != "RGB":
        question_img = question_img.convert("RGB")
    if gt_img.mode != "RGB":
        gt_img = gt_img.convert("RGB")

    # 添加边框
    def add_border(img):
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, img.width - 1, img.height - 1), outline=border_color, width=border_width)
        return img

    question_img = add_border(question_img)
    gt_img = add_border(gt_img)

    # 为标签预留空间
    label_height = label_font_size * 2 + 20  # 标签高度 + 边距
    
    # 创建带标签空间的图像
    def create_image_with_label(img, label_text):
        # 创建新图像，为标签预留空间
        new_width = img.width
        new_height = img.height + label_height
        new_img = Image.new("RGB", (new_width, new_height), (255, 255, 255))
        
        # 将原图像粘贴到下方
        new_img.paste(img, (0, label_height))
        
        # 添加标签到顶部
        return add_text_label(new_img, label_text, label_font_size * 2, "top")
    
    question_img_with_label = create_image_with_label(question_img, "QUESTION")
    gt_img_with_label = create_image_with_label(gt_img, "OPTIONS")

    # 决定拼接方向：根据图像形状判断
    # 横向矩形（宽>高）：上下拼接
    # 纵向矩形（高>宽）：左右拼接
    if question_img.width > question_img.height:
        # 横向矩形：上下拼接
        total_width = max(question_img_with_label.width, gt_img_with_label.width)
        total_height = question_img_with_label.height + gt_img_with_label.height + gap_width

        combined_img = Image.new("RGB", (total_width, total_height), (255, 255, 255))
        combined_img.paste(question_img_with_label, (0, 0))
        combined_img.paste(gt_img_with_label, (0, question_img_with_label.height + gap_width))
    else:
        # 纵向矩形：左右拼接
        total_width = question_img_with_label.width + gt_img_with_label.width + gap_width
        total_height = max(question_img_with_label.height, gt_img_with_label.height)

        combined_img = Image.new("RGB", (total_width, total_height), (255, 255, 255))
        combined_img.paste(question_img_with_label, (0, 0))
        combined_img.paste(gt_img_with_label, (question_img_with_label.width + gap_width, 0))

    # 保存结果
    combined_img.save(output_path)
    print(f"输出保存为: {output_path}")
    return output_path


# Function to load, shuffle and split the dataset
def process_dataset(input_file, train_output_file, test_output_file, train_ratio=0.7):
    # Read the JSON file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for d in data:
        if d["class"] == "raven":
            image_id = d["imgs"][0].split("/")[-1].split(".")[0]
            input_dir = "/".join(d["imgs"][0].split("/")[:-1])
            output_dir = input_dir
            output_path = raven_cat(image_id, input_dir, output_dir)
            d["imgs"] = [output_path]
            d["option"] = "1,2,3,4,5,6,7,8"
        elif "A、A" in d["option"]:
            d["option"] = "A,B,C,D"

    # Shuffle the dataset
    random.shuffle(data)

    # Split the dataset into training and testing sets (7:3 ratio)
    train_data, test_data = train_test_split(data, train_size=train_ratio, random_state=42)

    # Save training set
    with open(train_output_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    # Save testing set
    with open(test_output_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    return len(train_data), len(test_data)


# Example usage
if __name__ == "__main__":
    input_file = "VisuRiddles.json"
    train_output_file = "train_dataset.json"
    test_output_file = "test_dataset.json"

    train_size, test_size = process_dataset(input_file, train_output_file, test_output_file)
    print(f"Training set size: {train_size}")
    print(f"Testing set size: {test_size}")