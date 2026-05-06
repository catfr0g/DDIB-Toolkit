# Валидация робастности моделей DDIB

Этот модуль предоставляет инструменты для оценки робастности моделей, обученных с использованием DDIB (Deep Data Information Bottleneck).

## Методы оценки робастности

### 1. CIFAR-10-C / ImageNet-C

**15 типов искажений × 5 уровней серьезности:**

| Категория | Искажения |
|-----------|-----------|
| **Noise** | gaussian_noise, shot_noise, impulse_noise |
| **Blur** | defocus_blur, glass_blur, motion_blur, zoom_blur |
| **Weather** | snow, frost, fog |
| **Digital** | brightness, contrast, elastic_transform, pixelate, jpeg_compression |

**Метрики:**
- **CE (Corruption Error)**: Ошибка на каждом типе искажения
- **mCE (mean Corruption Error)**: Средняя ошибка по всем искажениям, нормализованная относительно референсной модели (AlexNet)

### 2. Adversarial Examples (PGD Attack)

**Параметры PGD-атаки:**
- ε (epsilon) = 8/255 ≈ 0.031 (максимальный размер пертурбации)
- α (alpha) = 2/255 ≈ 0.008 (размер шага)
- Iterations = 10
- Norm = L∞

**Метрики:**
- **Clean Accuracy**: Точность на чистых данных
- **Adversarial Accuracy**: Точность на adversarial примерах
- **Robustness Gap**: Разница между clean и adversarial accuracy
- **mAA (mean Adversarial Accuracy)**: Средняя точность при разных атаках
- **Relative Robustness**: Отношение adversarial accuracy к clean accuracy

## Установка

```bash
# Установить зависимости
make requirements

# Или через uv
uv sync
```

## Быстрый старт

### 0. Подготовка данных CIFAR-10-C

```bash
# Скачать и подготовить CIFAR-10-C
make cifar10c-data

# Или напрямую
python -m src.experiments.robustness.prepare_data \
    --data-dir data/processed

# Проверить целостность данных
make verify-cifar10c
```

**Примечание:** CIFAR-10-C будет загружен из [Zenodo](https://zenodo.org/records/2535967) (~360 MB) и организован в структуру:

```
data/processed/CIFAR-10-C/
    gaussian_noise/
        1/
            images.npy
            labels.npy
        2/
            ...
    ...
```

### 1. Валидация готовой модели

```bash
# Валидация одной модели
make robustness-validate

# Или напрямую
python -m src.experiments.robustness.validate \
    --model-path models/best_model.pt \
    --model-arch vgg11 \
    --bottleneck-width 2048 \
    --data-dir data/processed \
    --output-dir results/robustness
```

### 2. Обучение и валидация с нуля

```bash
# Обучить и оценить лучшую конфигурацию
make robustness-validate-all

# Или напрямую
python -m src.experiments.robustness.validate train-and-evaluate \
    --config vgg11_best \
    --data-dir data/processed \
    --output-dir results/robustness \
    --num-epochs 100
```

### 3. Анализ результатов

```bash
# Запустить ноутбук анализа
make robustness-analysis

# Или открыть ноутбук вручную
jupyter notebook notebooks/robustness_analysis.ipynb
```

## Доступные конфигурации

В модуле `src/experiments/robustness/config.py` определены лучшие конфигурации из экспериментов:

| Название | Модель | Bottleneck | Beta | Test Accuracy |
|----------|--------|------------|------|---------------|
| `vgg11_best` | VGG11 | 2048 | 1e-08 | 85.92% |
| `vgg11_alt` | VGG11 | 2048 | 1e-06 | 85.75% |
| `resnet18_best` | ResNet18 | 2048 | 1e-06 | 84.65% |
| `resnet18_alt` | ResNet18 | 2048 | 1e-07 | 84.00% |
| `efficientnet_b0_best` | EfficientNet-B0 | 16 | 1e-06 | 78.31% |

## Структура модуля

```
src/experiments/robustness/
├── __init__.py          # Экспорт основных функций
├── config.py            # Конфигурации лучших моделей
├── imagenet_c.py        # Загрузчик CIFAR-10-C / ImageNet-C
├── pgd_attack.py        # PGD-атака для adversarial examples
├── metrics.py           # Метрики робастности (mCE, mAA)
└── validate.py          # Основной скрипт валидации
```

## Использование в коде

### Оценка на коррумпианных данных

```python
from src.experiments.robustness import (
    create_cifar10_c_dataloader,
    calculate_robustness_metrics,
    print_robustness_report,
)

# Создать dataloader для CIFAR-10-C
dataloader = create_cifar10_c_dataloader(
    data_dir=Path('data/processed/CIFAR-10-C'),
    corruption_types=['gaussian_noise', 'shot_noise'],
    severity_levels=[1, 2, 3, 4, 5],
    batch_size=64,
)

# Оценить модель
accuracies = {}
for corruption_type in ['gaussian_noise', 'shot_noise']:
    accuracies[corruption_type] = {}
    for severity in range(1, 6):
        acc = evaluate_on_severity(model, dataloader, corruption_type, severity)
        accuracies[corruption_type][severity] = acc

# Рассчитать метрики
metrics = calculate_robustness_metrics(corruption_accuracies=accuracies)
print_robustness_report(metrics)
```

### Оценка на adversarial примерах

```python
from src.experiments.robustness import PGDAttack, evaluate_adversarial_robustness

# Создать PGD-атаку
attack = PGDAttack(
    model=model,
    epsilon=8/255,
    alpha=2/255,
    iterations=10,
)

# Сгенерировать adversarial примеры и оценить
adv_accuracy, metrics = attack.attack_accuracy(
    images=test_images,
    labels=test_labels,
    batch_size=32,
)

# Или использовать готовую функцию
results = evaluate_adversarial_robustness(
    model=model,
    images=test_images,
    labels=test_labels,
    epsilon=8/255,
    alpha=2/255,
    iterations=10,
)
```

## Интерпретация результатов

### Хорошие показатели робастности

| Метрика | Хорошо | Средне | Плохо |
|---------|--------|--------|-------|
| mCE | < 0.5 | 0.5-0.7 | > 0.7 |
| mAA | > 0.6 | 0.4-0.6 | < 0.4 |
| Robustness Gap | < 0.1 | 0.1-0.2 | > 0.2 |
| Relative Robustness | > 0.8 | 0.6-0.8 | < 0.6 |

### Визуализация

Ноутбук `notebooks/robustness_analysis.ipynb` создаёт следующие визуализации:

1. **mCE vs Clean Accuracy** - Scatter plot для сравнения моделей
2. **Clean vs Adversarial Accuracy** - Bar chart по моделям
3. **Corruption Error Heatmap** - Тепловая карта ошибок по типам искажений
4. **Accuracy by Severity** - Линейные графики ухудшения точности
5. **Error by Category** - Сравнение по категориям искажений

## Ссылки

- [ImageNet-C Paper](https://arxiv.org/abs/1903.12261) - Benchmarking Neural Network Robustness to Common Corruptions and Perturbations
- [PGD Paper](https://arxiv.org/abs/1706.06083) - Towards Deep Learning Models Resistant to Adversarial Attacks
- [CIFAR-10-C](https://github.com/hendrycks/robustness) - Репозиторий с данными

## Примеры команд

```bash
# Пропустить оценку на коррумпианных данных (только PGD)
python -m src.experiments.robustness.validate \
    --model-path models/model.pt \
    --skip-corruptions

# Пропустить PGD (только коррумпианные данные)
python -m src.experiments.robustness.validate \
    --model-path models/model.pt \
    --skip-pgd

# Оценить конкретную конфигурацию
python -m src.experiments.robustness.validate train-and-evaluate \
    --config resnet18_best \
    --num-epochs 150 \
    --seed 42
```
