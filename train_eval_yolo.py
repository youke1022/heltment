import os
from ultralytics import YOLO

# 设置中文字体支持
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

class YOLOModelTrainer:
    def __init__(self, data_yaml, model_type='yolov8n.pt', epochs=100, imgsz=640):
        """初始化YOLO模型训练器"""
        self.data_yaml = data_yaml
        self.model_type = model_type
        self.epochs = epochs
        self.imgsz = imgsz
        self.model = None
        self.results = None
        
    def train(self):
        """训练YOLO模型"""
        print(f"开始训练模型: {self.model_type}")
        print(f"数据集配置: {self.data_yaml}")
        print(f"训练参数: epochs={self.epochs}, imgsz={self.imgsz}")
        
        # 加载模型
        self.model = YOLO(self.model_type)
        
        # 开始训练
        self.results = self.model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=16,  # 减小批量大小适应4GB显存
            device=0 if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',  # 更可靠的GPU检测
            workers=4 if os.name == 'nt' else 8,  # Windows启用4个工作线程
            name='helmet_detection',
            amp=True,
            patience=5,  # 早停机制
            augment=False  # 禁用高级数据增强
        )
        
        print("训练完成!")
        return self.results
        
    def evaluate(self, model_path=None):
        """评估模型性能"""
        if not model_path and not self.model:
            raise ValueError("请先训练模型或提供模型路径")
            
        # 如果提供了模型路径，则加载该模型
        if model_path:
            print(f"加载模型进行评估: {model_path}")
            self.model = YOLO(model_path)
        
        print("开始评估模型性能...")
        metrics = self.model.val()
        
        # 打印评估结果
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP75: {metrics.box.map75:.4f}")
        print(f"精确率: {metrics.box.mp:.4f}")
        print(f"召回率: {metrics.box.mr:.4f}")
        
        return metrics
        
    def predict(self, img_path, model_path=None, save=True):
        """使用模型进行预测"""
        if not model_path and not self.model:
            raise ValueError("请先训练模型或提供模型路径")
            
        # 如果提供了模型路径，则加载该模型
        if model_path:
            print(f"加载模型进行预测: {model_path}")
            self.model = YOLO(model_path)
        
        print(f"对图像进行预测: {img_path}")
        results = self.model.predict(
            source=img_path,
            save=save,
            imgsz=self.imgsz,
            conf=0.25,
            iou=0.5
        )
        
        if save:
            print(f"预测结果已保存到: {results[0].save_dir}")
        
        return results

if __name__ == '__main__':
    # 直接设置训练参数，不再需要命令行参数
    data_yaml = r'c:\Users\15110\Desktop\安全帽\data.yaml'
    model_type = 'yolov8n.pt'
    epochs = 30
    imgsz = 320
    
    # 创建训练器实例
    trainer = YOLOModelTrainer(
        data_yaml=data_yaml,
        model_type=model_type,
        epochs=epochs,
        imgsz=imgsz
    )
    
    # 直接执行训练和评估
    print("开始执行训练和评估流程...")
    training_results = trainer.train()
    print("训练完成，开始使用训练结果进行评估...")
    evaluation_results = trainer.evaluate()
    print("训练和评估流程已完成！")