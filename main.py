from urlbert_trainer import URLBERTClassifierTrainer, ModelConfig
config = ModelConfig(
    urlbert_batch_size=64,  # 根据GPU内存调整
    validation_split=0.2,
    use_lightgbm=True
)

trainer = URLBERTClassifierTrainer(config)

results = trainer.train(
    json_file='training_data/labeled_urls.json',
    save_path='models/urlbert_classifier.pkl'
)

print(f"训练准确率: {results['train_accuracy']:.4f}")
print(f"验证准确率: {results['val_accuracy']:.4f}")
print(f"类别: {results['classes']}")