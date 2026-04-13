# 游戏视频处理步骤

用Qwen3-VL-32B的大模型对游戏画面进行描述之后，形成一个描述性的数据集，然后对这个数据集进行一些列处理，最终形成游戏AI助手的训练数据

## 功能特点

- 支持从视频中提取帧并生成详细的操作描述
- 支持基于字幕时间戳的片段提取或固定时长片段
- 生成适合微调训练的数据格式
- 支持批量处理多个视频

## 环境要求

### 1. 处理步骤

```bash
python pair_clips_with_previous.py   # 跟前一段视频形成数据对
```

### 3. 硬件要求
- GPU: 推荐使用 NVIDIA GPU，至少 16GB 显存（用于 Qwen2-VL-8B）
- 如果显存不足，可以考虑使用 Qwen2-VL-2B-Instruct 等更小的模型

## 使用方法

### 方法1: 完整版本（推荐用于生产环境）

完整版本支持基于字幕片段的智能分割和固定时长分割两种模式。

```bash
# 使用字幕片段（从JSON文件中提取）
python describe_videos_with_qwen.py \
    --video_dir processed_bilibili_cs2 \
    --output_file video_descriptions.jsonl \
    --model_name Qwen/Qwen2-VL-8B-Instruct \
    --use_subtitle_segments \
    --min_segment_duration 5.0 \
    --max_segment_duration 30.0 \
    --max_videos 10

# 使用固定时长片段
python describe_videos_with_qwen.py \
    --video_dir processed_bilibili_cs2 \
    --output_file video_descriptions.jsonl \
    --model_name Qwen/Qwen2-VL-8B-Instruct \
    --clip_duration 10 \
    --clip_overlap 2 \
    --max_videos 10
```

### 方法2: 简化版本（推荐用于快速测试）

简化版本从每个视频中均匀提取几帧，快速生成描述。

```bash
python describe_videos_simple.py \
    --video_dir processed_bilibili_cs2 \
    --output_file video_descriptions_simple.jsonl \
    --model_name Qwen/Qwen2-VL-8B-Instruct \
    --max_videos 5 \
    --frames_per_segment 4
```

## 参数说明

### 完整版本参数

- `--video_dir`: 视频文件目录（默认: `processed_bilibili_cs2`）
- `--output_file`: 输出描述文件路径（默认: `video_descriptions.jsonl`）
- `--model_name`: 模型名称或路径（默认: `Qwen/Qwen2-VL-8B-Instruct`）
- `--device`: 运行设备，`cuda` 或 `cpu`（默认: 自动检测）
- `--max_videos`: 最多处理多少个视频（用于测试，默认: 无限制）
- `--clip_duration`: 每个视频片段的时长（秒，默认: 10）
- `--clip_overlap`: 片段之间的重叠时长（秒，默认: 2）
- `--fps`: 从视频中提取帧的帧率（每秒多少帧，默认: 1）
- `--max_frames`: 每个片段最多使用多少帧（默认: 8）
- `--use_subtitle_segments`: 使用JSON中的字幕片段作为视频片段
- `--min_segment_duration`: 最小片段时长（秒，默认: 5.0）
- `--max_segment_duration`: 最大片段时长（秒，默认: 30.0）

### 简化版本参数

- `--video_dir`: 视频文件目录
- `--output_file`: 输出描述文件路径
- `--model_name`: 模型名称或路径
- `--device`: 运行设备
- `--max_videos`: 最多处理多少个视频（默认: 5）
- `--frames_per_segment`: 每个片段提取多少帧（默认: 4）
- `--segment_duration`: 每个片段的时长（秒，默认: 10）

## 输出格式

生成的 JSONL 文件每行包含一个视频片段的描述：

```json
{
  "video_id": "av9429688_",
  "video_path": "/path/to/video.mp4",
  "start_time": 2.7,
  "end_time": 10.715,
  "duration": 8.015,
  "subtitle": "能打的人自下陷阱，",
  "description": "玩家正在执行一个战术操作...",
  "segment_index": 0
}
```

## 模型选择

### 可用的模型

1. **Qwen2-VL-8B-Instruct** (推荐)
   - 模型名称: `Qwen/Qwen2-VL-8B-Instruct`
   - 显存需求: ~16GB
   - 性能: 优秀

2. **Qwen2-VL-2B-Instruct** (显存不足时使用)
   - 模型名称: `Qwen/Qwen2-VL-2B-Instruct`
   - 显存需求: ~6GB
   - 性能: 良好

3. **Qwen3-VL-8B** (如果已发布)
   - 模型名称: `Qwen/Qwen3-VL-8B-Instruct`
   - 显存需求: ~16GB
   - 性能: 更优秀

## 使用建议

### 1. 首次使用
建议先用简化版本测试 1-2 个视频，确保环境配置正确：

```bash
python describe_videos_simple.py --max_videos 1
```

### 2. 生产环境
使用完整版本，启用字幕片段模式，可以获得更准确的片段分割：

```bash
python describe_videos_with_qwen.py \
    --use_subtitle_segments \
    --min_segment_duration 5.0 \
    --max_segment_duration 30.0
```

### 3. 显存优化
如果遇到显存不足的问题：
- 使用更小的模型（如 Qwen2-VL-2B-Instruct）
- 减少 `--max_frames` 参数（如改为 4）
- 减少 `--fps` 参数（如改为 0.5）

### 4. 批量处理
对于大量视频，建议：
- 分批处理，每次处理 10-20 个视频
- 使用 `--max_videos` 参数控制每次处理的数量
- 可以编写脚本循环调用，每次处理不同的视频范围

## 后续步骤

生成描述后，可以：

1. **数据清洗**: 检查生成的描述质量，过滤低质量数据
2. **格式转换**: 将描述转换为适合微调 4B 模型的格式
3. **微调模型**: 使用生成的数据对 4B 模型进行微调
4. **评估测试**: 测试微调后的模型在游戏场景中的表现

## 故障排除

### 问题1: 模型加载失败
- 确保 transformers 版本 >= 4.37.0
- 检查网络连接（首次下载模型需要）
- 如果使用本地模型，检查路径是否正确

### 问题2: 显存不足
- 使用更小的模型
- 减少 `--max_frames` 参数
- 使用 CPU 模式（会很慢）

### 问题3: 视频读取失败
- 确保安装了 opencv-python
- 检查视频文件是否损坏
- 确保视频格式支持（MP4, AVI 等）

### 问题4: 描述生成失败
- 检查模型是否正确加载
- 查看错误日志
- 尝试减少输入帧数

## 示例输出

描述示例：
```
玩家正在执行一个战术操作。首先，玩家选择了地图下方的位置，这是一个常见的防守点位。
玩家使用了陷阱道具，这是一个战术性的选择，可以限制敌人的移动路径。在放置陷阱后，
玩家迅速移动到三楼位置，这是一个高地的优势位置，可以提供更好的视野和射击角度。
玩家的移动非常流畅，展现了良好的地图理解和位置意识。这种操作组合体现了高水平的
战术思维和执行力。
```

## 注意事项

1. 视频处理需要大量计算资源，建议使用 GPU
2. 处理大量视频需要较长时间，建议分批处理
3. 生成的描述质量取决于模型能力和视频质量
4. 建议对生成的描述进行人工审核和清洗

## 许可证

请遵循 Qwen 模型的使用许可协议。



