# -*- coding: utf-8 -*-
"""
SAM3 Image Segmentation Studio
"""

import os
import io
import zipfile
import torch
import warnings
import numpy as np
import cv2
import gradio as gr
from PIL import Image
from datetime import datetime

# ==================== 环境配置 ====================
os.environ["USE_PERFLIB"] = "0"

# Patch perflib
try:
    import sam3.perflib.connected_components as cc_lib
    original_cc_cpu = cc_lib.connected_components_cpu
    def connected_components_cpu_safe(input_tensor):
        if input_tensor.numel() == 0 or input_tensor.shape[0] == 0:
            return torch.zeros_like(input_tensor), torch.zeros_like(input_tensor)
        return original_cc_cpu(input_tensor)
    cc_lib.connected_components = connected_components_cpu_safe
    print("[OK] Patched connected_components")
except Exception as e:
    print(f"[WARN] Could not patch connected_components: {e}")

try:
    import sam3.perflib.nms as nms_lib
    def nms_cpu_wrapper(ious, scores, iou_threshold):
        device = scores.device
        scores_cpu = scores.cpu() if scores.is_cuda else scores
        ious_cpu = ious.cpu() if ious.is_cuda else ious
        keep = []
        remaining = list(range(len(scores_cpu)))
        while remaining:
            idx = max(remaining, key=lambda i: scores_cpu[i].item())
            keep.append(idx)
            remaining.remove(idx)
            remaining = [i for i in remaining if ious_cpu[idx, i].item() < iou_threshold]
        return torch.tensor(keep, dtype=torch.long, device=device)
    nms_lib.generic_nms = nms_cpu_wrapper
    print("[OK] Patched nms")
except Exception as e:
    print(f"[WARN] Could not patch nms: {e}")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

warnings.filterwarnings("ignore")

# ==================== 配置 ====================
MODEL_PATH = "./model"
CHECKPOINT_NAME = "sam3.pt"
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局模型
image_processor = None

# 分割可视化颜色
COLORS = [
    (99, 102, 241),   # 靛蓝
    (236, 72, 153),   # 粉红
    (34, 197, 94),    # 绿色
    (251, 146, 60),   # 橙色
    (139, 92, 246),   # 紫色
    (20, 184, 166),   # 青色
    (239, 68, 68),    # 红色
    (250, 204, 21),   # 黄色
]

if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()


def initialize_model():
    """初始化模型"""
    global image_processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    
    checkpoint_path = os.path.join(MODEL_PATH, CHECKPOINT_NAME)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")
    
    print("[INFO] Loading SAM3 model...")
    image_model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        device=device,
        load_from_HF=False
    )
    image_processor = Sam3Processor(image_model)
    print("[OK] Model loaded!")


def visualize_masks(image: np.ndarray, masks: np.ndarray, alpha: float = 0.4):
    """可视化分割结果"""
    result = image.copy()
    
    for i, mask in enumerate(masks):
        color = COLORS[i % len(COLORS)]
        mask_bool = mask > 0.5
        
        # 叠加颜色
        for c in range(3):
            result[:, :, c] = np.where(
                mask_bool,
                result[:, :, c] * (1 - alpha) + color[c] * alpha,
                result[:, :, c]
            )
        
        # 绘制轮廓
        contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)
    
    return result.astype(np.uint8)


def extract_objects(image: np.ndarray, masks: np.ndarray):
    """提取分割对象（透明背景）"""
    objects = []
    
    for mask in masks:
        mask_bool = mask > 0.5
        coords = np.where(mask_bool)
        
        if len(coords[0]) == 0:
            continue
        
        # 创建 RGBA
        rgba = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
        rgba[:, :, :3] = image
        rgba[:, :, 3] = (mask_bool * 255).astype(np.uint8)
        
        # 裁剪
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        cropped = rgba[y_min:y_max+1, x_min:x_max+1]
        
        objects.append(Image.fromarray(cropped, 'RGBA'))
    
    return objects


def segment(image, prompt, confidence, max_objects, min_area):
    """执行分割"""
    global image_processor
    
    if image is None:
        gr.Warning("Please upload an image")
        return None, [], "Please upload an image", []
    
    if not prompt or prompt.strip() == "":
        gr.Warning("Please enter a prompt")
        return None, [], "Please enter a prompt", []
    
    try:
        # 处理图像
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = Image.fromarray(image).convert("RGB")
        
        image_np = np.array(pil_image)
        
        # 分割
        inference_state = image_processor.set_image(pil_image)
        
        try:
            inference_state = image_processor.set_confidence_threshold(confidence, inference_state)
        except:
            pass
        
        outputs = image_processor.set_text_prompt(prompt=prompt.strip(), state=inference_state)
        
        # 处理结果
        masks = outputs["masks"]
        scores = outputs["scores"]
        
        if isinstance(masks, torch.Tensor):
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            masks_np = masks.cpu().numpy()
        else:
            masks_np = np.array(masks)
        
        if isinstance(scores, torch.Tensor):
            scores_np = scores.float().cpu().numpy()
        else:
            scores_np = np.array(scores)
        
        # 过滤小面积（仅当 min_area > 0 时）
        if min_area > 0:
            valid = [(masks_np[i], scores_np[i]) for i in range(len(masks_np)) if (masks_np[i] > 0.5).sum() >= min_area]
        else:
            valid = [(masks_np[i], scores_np[i]) for i in range(len(masks_np))]
        
        if not valid:
            return image_np, [], "No objects detected", []
        
        masks_np = np.array([v[0] for v in valid])
        scores_np = np.array([v[1] for v in valid])
        
        # 限制数量（仅当 max_objects > 0 时）
        if max_objects > 0 and len(masks_np) > max_objects:
            indices = np.argsort(scores_np)[::-1][:max_objects]
            masks_np = masks_np[indices]
            scores_np = scores_np[indices]
        
        # 可视化
        result = visualize_masks(image_np, masks_np)
        
        # 提取对象
        objects = extract_objects(image_np, masks_np)
        
        # 保存对象
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        object_paths = []
        for i, obj in enumerate(objects):
            path = os.path.join(OUTPUT_DIR, f"object_{timestamp}_{i}.png")
            obj.save(path, "PNG")
            object_paths.append(path)
        
        status = f"[OK] Found {len(masks_np)} objects | Avg confidence: {scores_np.mean():.1%}"
        
        return result, object_paths, status, object_paths
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, [], f"[ERROR] {str(e)}", []


def download_objects(object_paths):
    """打包下载"""
    if not object_paths:
        gr.Warning("No objects to download")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(OUTPUT_DIR, f"objects_{timestamp}.zip")
    
    count = 0
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for i, path in enumerate(object_paths):
            if path and os.path.exists(path):
                zf.write(path, f"object_{i+1}.png")
                count += 1
    
    if count == 0:
        gr.Warning("No valid objects found")
        return None
    
    return zip_path


# 自定义 CSS
CUSTOM_CSS = """
.title-text {
    text-align: center !important;
    font-size: 2.2em !important;
    font-weight: 600 !important;
    margin-bottom: 0.2em !important;
}
.subtitle-text {
    text-align: center !important;
    color: #666 !important;
    margin-bottom: 1.5em !important;
}
"""


def create_app():
    """创建应用"""
    
    theme = gr.themes.Ocean(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )
    
    with gr.Blocks(theme=theme, title="SAM3 Segmentation", css=CUSTOM_CSS) as app:
        
        # 标题（居中）
        gr.Markdown("<h1 class='title-text'>SAM3 Image Segmentation</h1>", elem_classes="title-text")
        gr.Markdown("<p class='subtitle-text'>Upload an image and describe what you want to segment</p>")
        
        with gr.Row(equal_height=True):
            # 左侧 - 输入
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    height=400
                )
                
                prompt_input = gr.Textbox(
                    label="What to segment",
                    placeholder="e.g., cat, dog, person, car, bottle...",
                    lines=1
                )
                
                with gr.Accordion("Advanced Options", open=False):
                    confidence = gr.Slider(0.1, 1.0, 0.5, step=0.05, label="Confidence Threshold")
                    max_objects = gr.Slider(0, 50, 0, step=1, label="Max Objects (0 = unlimited)")
                    min_area = gr.Slider(0, 10000, 0, step=100, label="Min Area in pixels (0 = no filter)")
                
                segment_btn = gr.Button("Start Segmentation", variant="primary", size="lg")
            
            # 右侧 - 结果
            with gr.Column(scale=1):
                result_image = gr.Image(
                    label="Segmentation Result",
                    height=400,
                    interactive=False
                )
                
                status_text = gr.Textbox(label="Status", interactive=False)
        
        # 存储对象路径
        object_paths_state = gr.State([])
        
        # 底部 - 分割对象
        gr.Markdown("### Extracted Objects")
        
        with gr.Row():
            objects_gallery = gr.Gallery(
                label="",
                columns=6,
                height=180,
                object_fit="contain",
                show_label=False
            )
        
        with gr.Row():
            download_btn = gr.Button("Download All Objects (ZIP)", size="sm")
        
        download_file = gr.File(label="Download", visible=True)
        
        # 使用说明
        with gr.Accordion("Help", open=False):
            gr.Markdown(
                """
                **How to use:**
                1. Upload an image
                2. Enter what you want to segment (English works best)
                3. Click "Start Segmentation"
                4. Download the extracted objects (transparent PNG)
                
                **Prompt examples:**
                - Single object: `cat`, `dog`, `person`, `car`
                - Multiple objects: `cat and dog`, `all people`
                - With attributes: `red car`, `black cat`, `tall building`
                
                **Parameters:**
                - **Confidence Threshold**: Higher = stricter, recommended 0.3-0.7
                - **Max Objects**: Limit number of results, 0 = no limit
                - **Min Area**: Filter out small fragments, 0 = no filter
                """
            )
        
        # 绑定事件
        segment_btn.click(
            fn=segment,
            inputs=[image_input, prompt_input, confidence, max_objects, min_area],
            outputs=[result_image, objects_gallery, status_text, object_paths_state]
        )
        
        download_btn.click(
            fn=download_objects,
            inputs=[object_paths_state],
            outputs=[download_file]
        )
    
    return app


if __name__ == "__main__":
    print("=" * 50)
    print("SAM3 Image Segmentation Studio")
    print("=" * 50)
    
    initialize_model()
    
    app = create_app()
    app.launch(
        share=False
    )
