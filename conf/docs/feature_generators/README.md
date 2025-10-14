# Справочник по Генераторам Признаков

Этот раздел документации является полным справочником по всем доступным в фреймворке генераторам признаков.

Все генераторы — это Python-классы, наследующие от `FeatureGenerator`, которые используются в конвейерах в `conf/feature_engineering/`.

## 📚 Как пользоваться этим справочником

Генераторы сгруппированы по типу данных, с которыми они работают. Выберите интересующую вас категорию, чтобы перейти к детальному описанию доступных техник.

---

### **1. 🔢 Обработка Числовых Признаков (`Numerical`)**

Признаки для преобразования, масштабирования и дискретизации непрерывных числовых данных.

*   [**Масштабирование (`scaling`)**](./feature_generators/numerical/scaling.md): `StandardScaler`, `MinMaxScaler`, `RobustScaler`.
*   [**Математические преобразования (`transformations`):**](./feature_generators/numerical/transformations.md) `LogTransformer`, `BoxCoxTransformer`.
*   [**Дискретизация (`binning`):**](./feature_generators/numerical/binning.md) `QuantileBinner`, `DecisionTreeBinner`.
*   [**Флаги и Индикаторы (`flags`):**](./feature_generators/numerical/flags.md) `IsNullIndicator`, `OutlierIndicator`.

---

### **2. 🔠 Обработка Категориальных Признаков (`Categorical`)**

Признаки для преобразования текстовых или целочисленных категорий в числовой формат.

> ➡️ **[Перейти к детальному описанию категориальных генераторов](./feature_generators/categorical/README.md)**

*   [**Номинальное кодирование (`nominal`)**](./feature_generators/categorical/nominal.md): `OneHotEncoder`, `CountFrequencyEncoder`.
*   [**Порядковое кодирование (`ordinal`)**](./feature_generators/categorical/ordinal.md): `OrdinalEncoderGenerator`.
*   [**Кодирование на основе таргета (`target_based`)**](./feature_generators/categorical/target_based.md): `TargetEncoder`, `WoEEncoder`.
*   [**Комбинирование (`combination`)**](./feature_generators/categorical/combination.md): `RareCategoryCombiner`.

---

### **3. 📝 Обработка Текстовых Признаков (`Text`)**

Признаки для преобразования сырого текста в числовые представления.

> ➡️ **[Перейти к детальному описанию текстовых генераторов](./feature_generators/text/README.md)**

*   [**"Мешок слов" (`bow`)**](./feature_generators/text/bow.md): `TfidfVectorizer`, `CountVectorizer`.
*   [**Статистики (`statistics`)**](./feature_generators/text/statistics.md): `TextStatisticsGenerator` (длина текста, количество слов и т.д.).
*   [**Эмбеддинги (`embeddings`)**](./feature_generators/text/embeddings.md): `PretrainedEmbeddingGenerator` (GloVe, Word2Vec).
*   **Трансформеры (`transformer`):** `TransformerEmbeddingGenerator` (BERT, etc.).

---

### **4. 🌍 Высокоуровневые Признаки**

Генераторы, которые работают с несколькими признаками одновременно для создания сложных зависимостей.

*   **[Взаимодействия Признаков (Interactions)](./feature_generators/interaction.md):** Создание полиномиальных, частных и других взаимодействий.
*   **[Агрегации и Групповые Статистики](./feature_generators/aggregation.md):** Мощнейшие `groupby().agg()` операции.
*   **[Признаки из Даты и Времени](./feature_generators/datetime.md):** Извлечение компонентов (`день недели`, `месяц`) и вычисление разниц.

---

### **5. 🤖 Продвинутые Признаки на Основе Моделей (`Advanced`)**

Генераторы, которые используют вспомогательные модели для создания мета-признаков.

> ➡️ **[Перейти к детальному описанию продвинутых генераторов](./feature_generators/advanced/README.md)**

*   [**Кластеризация, PCA (`model_based`)**](./feature_generators/advanced/model_based.md): `KMeansFeatureGenerator`, `PCAGenerator`.
*   **Признаки из листьев деревьев (`model_based`):** `TreeLeafFeatureGenerator`.
*   [**Признаки на основе соседей (`neighbors`)**](./feature_generators/advanced/neighbors.md): `NearestNeighborsFeatureGenerator`.
*   **Self-Supervised (`autoencoder`):** `AutoencoderFeatureGenerator`.