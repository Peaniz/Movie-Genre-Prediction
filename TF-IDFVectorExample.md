
```markdown
# Minh họa TF-IDF với ví dụ mô tả phim

Để minh họa TF-IDF, chúng ta sử dụng **3 mô tả phim ngắn** (tiếng Anh) làm ví dụ. Ví dụ này sẽ cho thấy từng bước: tiền xử lý văn bản, chuyển đổi thành tokens, tính TF, IDF và sau đó tính TF-IDF.

---

## 1. Tiền xử lý văn bản

Trước khi tính TF-IDF, chúng ta **tiền xử lý** văn bản bao gồm:

- Chuyển đổi tất cả sang chữ thường (lowercase).
- Loại bỏ dấu câu và các ký tự không phải chữ (nếu cần).
- **Loại bỏ stop words** (những từ phổ thông không mang nhiều ý nghĩa như “a”, “the”, “is”,…).

Ví dụ:  
Với mô tả phim:

```

"A team of heroes fights to save the world. The world is threatened by an alien invasion."

````

Sau khi chuyển về chữ thường và bỏ dấu câu, ta có thể loại bỏ các stop word.

```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

desc = "A team of heroes fights to save the world."
text = desc.lower()
text = re.sub(r'[^\w\s]', '', text)  # loại bỏ dấu câu
stopwords = ENGLISH_STOP_WORDS
tokens = [w for w in text.split() if w not in stopwords]
print(tokens)  # ['team', 'heroes', 'fights', 'save', 'world']
````

---

## 2. Token hóa (Tokenization)

Sau khi tiền xử lý, mỗi mô tả phim được **token hóa** thành danh sách các từ (tokens).
Giả sử có 3 mô tả phim:

```python
docs = [
    "A team of heroes fights to save the world. The world is threatened by an alien invasion.",
    "A brilliant detective investigates a series of strange murders in a small town.",
    "A young wizard attends a school of magic where he fights a dark lord."
]
```

Áp dụng tiền xử lý và token hóa tương tự.

---

## 3. Tính TF (Term Frequency)

TF đo số lần một từ xuất hiện trong một tài liệu.

Sử dụng `TfidfVectorizer` để tính TF-IDF:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [docs[0], docs[1], docs[2]]
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus).toarray()
features = vectorizer.get_feature_names_out()

import pandas as pd
df_tfidf = pd.DataFrame(tfidf_matrix.T, index=features, columns=['Phim 1','Phim 2','Phim 3'])
print(df_tfidf)

Bảng TF mẫu:

| Từ     | Phim 1 | Phim 2 | Phim 3 |
| ------ | ------ | ------ | ------ |
| world  | 2      | 0      | 0      |
| fights | 1      | 0      | 1      |
| team   | 1      | 0      | 0      |
| heroes | 1      | 0      | 0      |
| alien  | 1      | 0      | 0      |
| ...    | ...    | ...    | ...    |

---

## 4. Tính IDF (Inverse Document Frequency)

Công thức:

```
idf(t) = log(N / df(t))
```

Trong đó:

* N là số tài liệu.
* df(t) là số tài liệu chứa từ t.

Ví dụ:

* Từ "fights" xuất hiện trong 2 tài liệu → IDF ≈ log(3/2) ≈ 0.4055
* Từ "world" chỉ xuất hiện trong 1 tài liệu → IDF ≈ log(3) ≈ 1.0986

---

## 5. Tính TF-IDF

TF-IDF = TF × IDF

Dùng `TfidfVectorizer`:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(norm=None, smooth_idf=False, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus).toarray()
features = vectorizer.get_feature_names_out()

df_tfidf = pd.DataFrame(tfidf_matrix.T, index=features, columns=['Phim 1','Phim 2','Phim 3'])
print(df_tfidf.round(4))
```

Bảng kết quả TF-IDF (giá trị làm tròn):

| Từ        | Phim 1 | Phim 2 | Phim 3 |
| --------- | ------ | ------ | ------ |
| world     | 4.1972 | 0      | 0      |
| fights    | 1.4055 | 0      | 1.4055 |
| heroes    | 2.0986 | 0      | 0      |
| alien     | 2.0986 | 0      | 0      |
| detective | 0      | 2.0986 | 0      |
| magic     | 0      | 0      | 2.0986 |
| wizard    | 0      | 0      | 2.0986 |
| ...       | ...    | ...    | ...    |

Giải thích:

* Từ “world” có TF=2, IDF≈2.0986 → TF-IDF = 2×2.0986 ≈ 4.1972
* Từ “fights” xuất hiện trong cả Phim 1 và Phim 3 → IDF thấp hơn.

---

## Tổng kết

TF-IDF giúp chọn lọc các từ quan trọng để biểu diễn nội dung tài liệu.
Trong bài toán phân loại mô tả phim theo thể loại, TF-IDF giúp mô hình tập trung vào các từ đặc trưng cho mỗi thể loại (như “wizard”, “magic”, “detective”, “alien”…).

```

---

Nếu bạn dùng trong Jupyter hoặc công cụ hiển thị Markdown, bảng và code đều sẽ được trình bày rõ ràng. Nếu bạn muốn mình xuất luôn file `.md`, hãy cho biết nhé!
```
