import os
import cv2
from ultralytics import YOLO
import time

# 设置中文字体支持
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

class YoloVideoInference:
    def __init__(self, model_path, device='auto'):
        """初始化YOLO视频推理器"""
        self.model_path = model_path
        self.device = device
        self.model = None
        
    def load_model(self):
        """加载YOLO模型"""
        print(f"正在加载模型: {self.model_path}")
        self.model = YOLO(self.model_path)
        if self.device != 'auto':
            self.model.to(self.device)
        print(f"模型加载成功，使用设备: {self.device if self.device != 'auto' else '自动选择'}")
        return self.model
    
    def process_video(self, video_path, conf_threshold=0.25, show_fps=True, save_output=True, output_path=None):
        """处理视频流，将检测结果保存为视频文件（不使用OpenCV窗口显示）"""
        if not self.model:
            self.load_model()
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: 分辨率={width}x{height}, FPS={fps:.2f}, 总帧数={total_frames}")
        
        # 设置输出视频 - 默认总是保存输出视频，因为无法显示
        if output_path is None:
            output_path = f"output_{os.path.basename(video_path)}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"检测结果将保存到: {output_path}")
        
        # 用于计算FPS
        prev_time = 0
        frame_count = 0
        total_processing_time = 0
        
        # 开始处理视频
        print("开始处理视频流...")
        print("处理过程中会显示进度信息...")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 计算FPS
                current_time = time.time()
                elapsed_time = current_time - prev_time
                prev_time = current_time
                inference_fps = 1 / elapsed_time if elapsed_time > 0 else 0
                total_processing_time += elapsed_time
                
                # 进行目标检测
                results = self.model(frame, conf=conf_threshold)
                
                # 绘制检测结果
                annotated_frame = results[0].plot()
                
                # 显示FPS
                if show_fps:
                    cv2.putText(annotated_frame, f"FPS: {inference_fps:.2f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示当前进度
                frame_count += 1
                progress = (frame_count / total_frames) * 100
                cv2.putText(annotated_frame, f"进度: {progress:.1f}%", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 保存输出视频
                out.write(annotated_frame)
                
                # 每处理100帧显示一次进度
                if frame_count % 100 == 0:
                    avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
                    remaining_time = (total_frames - frame_count) / avg_fps if avg_fps > 0 else 0
                    print(f"进度: {progress:.1f}% ({frame_count}/{total_frames}帧), 平均FPS: {avg_fps:.2f}, 剩余时间: {remaining_time/60:.1f}分钟")
        except KeyboardInterrupt:
            print("用户中断处理")
        except Exception as e:
            print(f"处理过程中出错: {e}")
        finally:
            # 释放资源
            cap.release()
            out.release()
            
            # 不调用destroyAllWindows，避免GUI错误
            print(f"视频处理完成! 共处理{frame_count}帧")
            print(f"检测结果已保存到: {output_path}")

if __name__ == '__main__':
    # 设置模型路径和视频路径
    model_path = r'c:\Users\15110\Desktop\安全帽\runs\detect\helmet_detection5\weights\best.pt'
    video_path = r'c:\Users\15110\Desktop\安全帽\test.mp4'  # 已经修改为实际视频路径
    
    # 创建视频推理器实例
    video_inferencer = YoloVideoInference(
        model_path=model_path,
        device='auto'  # 自动选择设备，也可以设置为'0'使用GPU，'cpu'使用CPU
    )
    
    # 处理视频流
    try:
        video_inferencer.process_video(
            video_path=video_path,
            conf_threshold=0.25,  # 置信度阈值
            show_fps=True,        # 在视频中显示FPS
            save_output=True,     # 保存输出视频（默认为True）
            output_path=r'c:\Users\15110\Desktop\安全帽\output_test.mp4'  # 指定输出视频路径
        )
    except Exception as e:
        print(f"处理视频时出错: {e}")
        print("请确保视频文件存在并且路径正确。")