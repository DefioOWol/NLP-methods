from collections import defaultdict
from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from src.extractor import extract_relations, facts_summary, NLP
from src.normalizer import normalize
from src.tokenizer import tokenize

PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

CMD_SENTIMENT = "<sentiment>"
CMD_RELATION = "<relation>"
CMD_TOPIC = "<topic>"
CMD_KEYWORDS = "<keywords>"
CMD_CONCLUSION = "<conclusion>"
CMD_PHRASE = "<phrase>"
CMD_SIMILAR = "<similar>"
CMD_FACTS = "<facts>"

COMMANDS = (
    CMD_SENTIMENT,
    CMD_RELATION,
    CMD_TOPIC,
    CMD_KEYWORDS,
    CMD_CONCLUSION,
    CMD_PHRASE,
    CMD_SIMILAR,
    CMD_FACTS,
)

TEMPLATES = {
    CMD_SENTIMENT: {
        "positive": "Текст выражает положительную оценку по теме {topic} (уверенность: {confidence}%).",
        "negative": "Текст содержит критическую оценку по теме {topic} (уверенность: {confidence}%).",
        "neutral": "Текст носит нейтрально-информационный характер по теме {topic} (уверенность: {confidence}%).",
    },
    CMD_RELATION: "Анализ связей фиксирует ключевые высказывания: {relation}.",
    CMD_TOPIC: "Тематика раскрывается через термины: {topic_terms}.",
    CMD_KEYWORDS: "Наибольший информационный вес имеют леммы: {keywords}.",
    CMD_CONCLUSION: "Семантическое окружение ассоциируется с понятиями: {associations}.",
    CMD_PHRASE: "В тексте выделяются устойчивые сочетания: {phrases}.",
    CMD_SIMILAR: "По смыслу текст близок к формулировке: «{similar}».",
    CMD_FACTS: "Из текста извлечены факты - {facts}.",
}


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def analyze_text(text: str) -> list[str]:
    return normalize(tokenize(text))


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        pad_idx: int,
        num_filters: int = 128,
        kernel_sizes: tuple[int, ...] = (2, 3, 4),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, num_filters, kernel_size=k)
                for k in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x).permute(0, 2, 1)
        pooled = [
            F.adaptive_max_pool1d(F.relu(conv(embedded)), 1).squeeze(2)
            for conv in self.convs
        ]
        features = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(features)


class EncoderLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        pad_idx: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(x))
        out, (hidden, cell) = self.lstm(embedded)

        batch_size = x.size(0)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)

        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)

        return out, hidden, cell


class DecoderLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        pad_idx: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim * 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(x))
        out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        out = self.fc(out.squeeze(1))
        return out, hidden, cell


class Seq2SeqLSTM(nn.Module):
    def __init__(
        self,
        encoder: EncoderLSTM,
        decoder: DecoderLSTM,
        pad_idx: int,
        bos_idx: int,
        eos_idx: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size, trg_len = src.size(0), trg.size(1)
        trg_vocab_size = self.decoder.fc.out_features

        out = torch.zeros(batch_size, trg_len, trg_vocab_size, device=src.device)
        _, hidden, cell = self.encoder(src)

        input_ = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            pred, hidden, cell = self.decoder(input_, hidden, cell)
            out[:, t, :] = pred

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = pred.argmax(dim=-1).unsqueeze(1)
            input_ = trg[:, t].unsqueeze(1) if teacher_force else top1

        return out

    def generate(self, src: torch.Tensor, *, max_len: int = 8) -> list[int]:
        _, hidden, cell = self.encoder(src)
        token = torch.tensor([[self.bos_idx]], device=src.device)
        generated = []

        for _ in range(max_len):
            pred, hidden, cell = self.decoder(token, hidden, cell)

            next_idx = int(pred.argmax(dim=-1).item())
            if next_idx in (self.eos_idx, self.pad_idx):
                break

            if next_idx not in generated:
                generated.append(next_idx)

            token = torch.tensor([[next_idx]], device=src.device)

        return generated


def build_text(
    commands: list[str],
    slots: dict[str, str],
    sentiment_class: str,
) -> str:
    sentences = []
    seen = set()

    for cmd in commands:
        if cmd in seen:
            continue
        seen.add(cmd)

        if cmd == CMD_SENTIMENT:
            tmpl_map = TEMPLATES[CMD_SENTIMENT]
            tmpl = tmpl_map.get(sentiment_class, tmpl_map["neutral"])
            rendered = tmpl.format(
                topic=slots.get("topic", "общая тематика"),
                confidence=slots.get("confidence", "100"),
            )
            sentences.append(rendered)

        elif cmd == CMD_RELATION:
            rel = slots.get("relation")
            if rel:
                rendered = TEMPLATES[CMD_RELATION].format(relation=rel)
                sentences.append(rendered)

        elif cmd == CMD_TOPIC:
            topic_terms = slots.get("topic_terms")
            if topic_terms:
                rendered = TEMPLATES[CMD_TOPIC].format(topic_terms=topic_terms)
                sentences.append(rendered)

        elif cmd == CMD_KEYWORDS:
            kw = slots.get("keywords")
            if kw:
                rendered = TEMPLATES[CMD_KEYWORDS].format(keywords=kw)
                sentences.append(rendered)

        elif cmd == CMD_CONCLUSION:
            assoc = slots.get("associations")
            if assoc:
                rendered = TEMPLATES[CMD_CONCLUSION].format(associations=assoc)
                sentences.append(rendered)

        elif cmd == CMD_PHRASE:
            phrases = slots.get("phrases")
            if phrases:
                rendered = TEMPLATES[CMD_PHRASE].format(phrases=phrases)
                sentences.append(rendered)

        elif cmd == CMD_SIMILAR:
            similar = slots.get("similar")
            if similar:
                rendered = TEMPLATES[CMD_SIMILAR].format(similar=similar)
                sentences.append(rendered)

        elif cmd == CMD_FACTS:
            facts = slots.get("facts")
            if facts:
                rendered = TEMPLATES[CMD_FACTS].format(facts=facts)
                sentences.append(rendered)

    if not sentences:
        sentences.append("Семантический анализ текста выполнен успешно.")

    doc = NLP(" ".join(sentences))

    cleaned_sentences = []
    for s in doc.sents:
        txt = s.text.strip()
        if txt:
            txt = txt[0].upper() + txt[1:]
            if not txt.endswith((".", "!", "?")):
                txt += "."
            cleaned_sentences.append(txt)

    return " ".join(cleaned_sentences)


class LSAModel:
    def __init__(
        self,
        n_topic_components: int = 20,
        vector_size: int = 100,
        window: int = 5,
        cnn_max_len: int = 100,
        cnn_filters: int = 128,
        cnn_kernel_sizes: tuple[int, ...] = (2, 3, 4),
        cnn_dropout: float = 0.3,
        seq2seq_hidden_dim: int = 128,
        seq2seq_num_layers: int = 2,
        seq2seq_dropout: float = 0.2,
        seq2seq_max_src_len: int = 40,
        seq2seq_max_trg_len: int = 10,
    ):
        self._vectorizer = TfidfVectorizer(
            lowercase=False,
            analyzer=analyze_text,
            max_df=0.95,
            min_df=2,
            max_features=5000,
            sublinear_tf=True,
        )
        self._svd = TruncatedSVD(n_components=n_topic_components)
        self._n_topic_components = n_topic_components
        self._vector_size = vector_size
        self._window = window

        self._cnn_max_len = cnn_max_len
        self._cnn_filters = cnn_filters
        self._cnn_kernel_sizes = cnn_kernel_sizes
        self._cnn_dropout = cnn_dropout
        self._cnn = None
        self._cnn_word2idx = {}
        self._label_encoder = LabelEncoder()

        self._seq2seq_hidden_dim = seq2seq_hidden_dim
        self._seq2seq_num_layers = seq2seq_num_layers
        self._seq2seq_dropout = seq2seq_dropout
        self._seq2seq_max_src_len = seq2seq_max_src_len
        self._seq2seq_max_trg_len = seq2seq_max_trg_len
        self._seq2seq = None

        self._seq_src2idx = {}
        self._seq_trg2idx = {}
        self._seq_idx2trg = {}

        self._word2vec = None
        self._doc2vec = None

        self._texts = None
        self._document_topics = None

    def fit(self, texts: list[str], *, epochs: int = 10) -> Self:
        self._texts = list(texts)

        tfidf = self._vectorizer.fit_transform(texts)
        self._document_topics = l2_normalize(self._svd.fit_transform(tfidf))

        tokenized = [analyze_text(t) for t in texts]
        self._word2vec = Word2Vec(
            sentences=tokenized,
            vector_size=self._vector_size,
            window=self._window,
            min_count=2,
            workers=1,
            sg=1,
            epochs=epochs,
        )

        tagged = [
            TaggedDocument(words=tokens, tags=[i])
            for i, tokens in enumerate(tokenized)
        ]
        self._doc2vec = Doc2Vec(
            documents=tagged,
            vector_size=self._vector_size,
            window=self._window,
            min_count=2,
            workers=1,
            dm=1,
            epochs=epochs,
        )

        return self

    def top_tfidf(self, text: str, *, k: int = 5) -> list[tuple[str, float]]:
        terms = self._vectorizer.get_feature_names_out()
        vector = self._vectorizer.transform([text])
        row = vector.toarray()[0]
        top_idx = np.argsort(row, descending=True)[:k]
        return [(terms[i], row[i]) for i in top_idx]

    def topic_terms(
        self, topic_id: int, *, k: int = 5
    ) -> list[tuple[str, float]]:
        terms = self._vectorizer.get_feature_names_out()
        weights = self._svd.components_[topic_id]
        top_idx = np.argsort(np.abs(weights), descending=True)[:k]
        return [(terms[i], weights[i]) for i in top_idx]

    def vectorize_topic(self, topic: str) -> np.ndarray:
        tfidf = self._vectorizer.transform([topic])
        return l2_normalize(self._svd.transform(tfidf))[0]

    def steer_topic(
        self, topic_vector: np.ndarray, weights: dict[int, float]
    ) -> np.ndarray:
        steered = topic_vector.copy()
        for topic_id, weight in weights.items():
            steered += self._svd.components_[topic_id] * weight
        return l2_normalize(steered.reshape(1, -1))[0]

    def topic_similar(
        self,
        topic_vector: np.ndarray,
        *,
        k: int = 5,
        preview_length: int | None = None,
    ) -> list[tuple[str, float]]:
        normalized = l2_normalize(topic_vector.reshape(1, -1))
        scores = cosine_similarity(normalized, self._document_topics)[0]

        results = []
        for i in np.argsort(scores, descending=True):
            results.append(
                (self.preview(self._texts[i], preview_length), scores[i])
            )
            if len(results) >= k:
                break

        return results

    def similar_words(self, word: str, *, k: int = 5) -> list[tuple[str, float]]:
        lemma = normalize(word)
        if lemma not in self._word2vec.wv:
            return []
        return [
            (word, score)
            for word, score in self._word2vec.wv.most_similar(
                positive=[lemma], topn=k
            )
        ]

    def vectorize_document(self, text: str) -> np.ndarray:
        tokens = analyze_text(text)
        return self._doc2vec.infer_vector(tokens, epochs=100)

    def similar_documents(
        self,
        text: str,
        *,
        k: int = 5,
        preview_length: int | None = None,
    ) -> list[tuple[str, float]]:
        vector = self.vectorize_document(text)
        return [
            (self.preview(self._texts[tag], preview_length), score)
            for tag, score in self._doc2vec.dv.most_similar(
                positive=[vector], topn=k
            )
        ]

    def pairwise_documents(self, texts: list[str]) -> np.ndarray:
        vectors = l2_normalize(
            np.vstack([self.vectorize_document(text) for text in texts])
        )
        return cosine_similarity(vectors)

    @staticmethod
    def preview(text: str, length: int | None) -> str:
        text = text.replace("\n", " ").strip()
        if length and len(text) > length:
            return text[:length] + "..."
        return text

    def fit_cnn(
        self,
        texts: list[str],
        labels: list[str],
        *,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
    ) -> Self:
        tokenized = [analyze_text(t) for t in texts]

        counts = defaultdict(int)
        for tokens in tokenized:
            for token in tokens:
                counts[token] += 1

        self._cnn_word2idx = {PAD_TOKEN: 0, UNKNOWN_TOKEN: 1}
        for token, count in sorted(counts.items()):
            if count >= 2:
                self._cnn_word2idx[token] = len(self._cnn_word2idx)

        encoded = self._encode_texts(tokenized)
        y = torch.tensor(
            self._label_encoder.fit_transform(labels), dtype=torch.long
        )
        loader = DataLoader(
            TensorDataset(encoded, y),
            batch_size=batch_size,
            shuffle=True,
        )

        self._cnn = TextCNN(
            vocab_size=len(self._cnn_word2idx),
            embed_dim=self._vector_size,
            num_classes=len(self._label_encoder.classes_),
            pad_idx=self._cnn_word2idx[PAD_TOKEN],
            num_filters=self._cnn_filters,
            kernel_sizes=self._cnn_kernel_sizes,
            dropout=self._cnn_dropout,
        )

        weights = self._cnn.embedding.weight.data
        for word, idx in self._cnn_word2idx.items():
            if word in self._word2vec.wv:
                weights[idx] = torch.tensor(
                    self._word2vec.wv[word], dtype=weights.dtype
                )

        optimizer = torch.optim.Adam(self._cnn.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self._cnn.train()
        for _ in range(epochs):
            for ids, labels in loader:
                optimizer.zero_grad()
                logits = self._cnn(ids)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        self._cnn.eval()
        return self

    @torch.inference_mode()
    def classify_corpus(
        self, texts: str | list[str]
    ) -> tuple[str, dict[str, float]] | list[tuple[str, dict[str, float]]]:
        single = isinstance(texts, str)
        docs = [texts] if single else texts
        encoded = self._encode_texts([analyze_text(doc) for doc in docs])

        logits = self._cnn(encoded)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        classes = [str(c) for c in self._label_encoder.classes_]
        results = []
        for i, pred_id in enumerate(probs.argmax(axis=1)):
            dist = {lbl: float(probs[i, j]) for j, lbl in enumerate(classes)}
            results.append((classes[pred_id], dist))

        return results[0] if single else results

    def _encode_texts(self, tokenized: list[list[str]]) -> torch.Tensor:
        pad = self._cnn_word2idx[PAD_TOKEN]
        unk = self._cnn_word2idx[UNKNOWN_TOKEN]

        rows = np.full((len(tokenized), self._cnn_max_len), pad, dtype=np.int32)

        for i, tokens in enumerate(tokenized):
            ids = [
                self._cnn_word2idx.get(token, unk)
                for token in tokens[:self._cnn_max_len]
            ]
            if not ids:
                ids = [unk]

            rows[i, :len(ids)] = ids

        return torch.from_numpy(rows)

    def _target_commands(self, text: str, sentiment: str | None = None) -> list[str]:
        extraction = extract_relations(text)
        tokens = analyze_text(text)

        commands = []
        if sentiment in ("positive", "negative"):
            commands.append(CMD_SENTIMENT)
        if extraction.relations:
            commands.append(CMD_RELATION)
        if len(tokens) >= 8:
            commands.append(CMD_TOPIC)
        if len(set(tokens)) >= 6:
            commands.append(CMD_KEYWORDS)
        if extraction.key_phrases:
            commands.append(CMD_PHRASE)
        if self._has_known_word(tokens):
            commands.append(CMD_CONCLUSION)
        if len(tokens) >= 12:
            commands.append(CMD_SIMILAR)
        if extraction.facts:
            commands.append(CMD_FACTS)
        if sentiment == "neutral":
            commands.append(CMD_SENTIMENT)

        return commands or [CMD_KEYWORDS]

    def _has_known_word(self, tokens: list[str]) -> bool:
        return any(token in self._word2vec.wv for token in tokens)

    def fit_generator(
        self,
        texts: list[str],
        labels: list[str] | None = None,
        *,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
    ) -> Self:
        tokenized = [analyze_text(t) for t in texts]

        counts = defaultdict(int)
        for tokens in tokenized:
            for token in tokens:
                counts[token] += 1

        self._seq_src2idx = {
            PAD_TOKEN: 0,
            UNKNOWN_TOKEN: 1,
            BOS_TOKEN: 2,
            EOS_TOKEN: 3,
        }
        for token, count in sorted(counts.items()):
            if count >= 2:
                self._seq_src2idx[token] = len(self._seq_src2idx)

        self._seq_trg2idx = {
            PAD_TOKEN: 0,
            UNKNOWN_TOKEN: 1,
            BOS_TOKEN: 2,
            EOS_TOKEN: 3,
            **{cmd: idx + 4 for idx, cmd in enumerate(COMMANDS)},
        }
        self._seq_idx2trg = {v: k for k, v in self._seq_trg2idx.items()}

        bos_trg = self._seq_trg2idx[BOS_TOKEN]
        eos_trg = self._seq_trg2idx[EOS_TOKEN]
        pad_trg = self._seq_trg2idx[PAD_TOKEN]

        X = np.full(
            (len(texts), self._seq2seq_max_src_len),
            self._seq_src2idx[PAD_TOKEN],
            dtype=np.int32,
        )
        for i, tokens in enumerate(tokenized):
            ids = [
                self._seq_src2idx.get(t, self._seq_src2idx[UNKNOWN_TOKEN])
                for t in tokens[:self._seq2seq_max_src_len]
            ]
            X[i, :len(ids)] = ids

        trg_seqs = []
        cmds_max_len = 0
        for i, text in enumerate(texts):
            sent = labels[i] if labels else None
            cmds = self._target_commands(text, sent)
            trg_seqs.append(cmds)
            cmds_max_len = max(len(cmds), cmds_max_len)

        trg_max_len = min(self._seq2seq_max_trg_len, cmds_max_len + 2)
        y = np.full((len(texts), trg_max_len), pad_trg, dtype=np.int32)

        for i, cmds in enumerate(trg_seqs):
            ids = (
                [bos_trg]
                + [self._seq_trg2idx[c] for c in cmds if c in self._seq_trg2idx]
                + [eos_trg]
            )
            ids = ids[:trg_max_len]
            y[i, :len(ids)] = ids

        X = torch.tensor(X, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=batch_size,
            shuffle=True,
        )

        encoder = EncoderLSTM(
            vocab_size=len(self._seq_src2idx),
            embed_dim=self._vector_size,
            pad_idx=pad_trg,
            hidden_dim=self._seq2seq_hidden_dim,
            num_layers=self._seq2seq_num_layers,
            dropout=self._seq2seq_dropout,
        )
        decoder = DecoderLSTM(
            vocab_size=len(self._seq_trg2idx),
            embed_dim=self._vector_size,
            pad_idx=pad_trg,
            hidden_dim=self._seq2seq_hidden_dim,
            num_layers=self._seq2seq_num_layers,
            dropout=self._seq2seq_dropout,
        )
        self._seq2seq = Seq2SeqLSTM(
            encoder=encoder,
            decoder=decoder,
            pad_idx=pad_trg,
            bos_idx=bos_trg,
            eos_idx=eos_trg,
        )

        optimizer = torch.optim.Adam(self._seq2seq.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_trg)

        self._seq2seq.train()
        for _ in range(epochs):
            for src, trg in loader:
                optimizer.zero_grad()
                out = self._seq2seq(src, trg)
                loss = criterion(
                    out[:, 1:].reshape(-1, out.shape[-1]),
                    trg[:, 1:].reshape(-1),
                )
                loss.backward()
                optimizer.step()

        self._seq2seq.eval()
        return self

    def plan_sequence(self, text: str) -> list[str]:
        unk = self._seq_src2idx[UNKNOWN_TOKEN]
        pad = self._seq_src2idx[PAD_TOKEN]
        tokens = analyze_text(text)

        ids = [
            self._seq_src2idx.get(token, unk)
            for token in tokens[:self._seq2seq_max_src_len]
        ]
        if not ids:
            ids = [unk]

        src_arr = np.full((1, self._seq2seq_max_src_len), pad, dtype=np.int32)
        src_arr[0, :len(ids)] = ids

        gen_ids = self._seq2seq.generate(
            torch.tensor(src_arr, dtype=torch.long),
            max_len=self._seq2seq_max_trg_len,
        )

        commands = [
            self._seq_idx2trg[i]
            for i in gen_ids
            if self._seq_idx2trg.get(i) in COMMANDS
        ]

        return commands

    def extract_semantic_summary(
        self, text: str, *, preview_length: int | None = None
    ) -> dict[str, Any]:
        commands = self.plan_sequence(text)
        sent_class, probs = self.classify_corpus(text)
        confidence = float(probs.get(sent_class, 1.0))

        tfidf_top = self.top_tfidf(text, k=5)
        kw_terms = [w for w, _ in tfidf_top]
        kw_str = ", ".join(kw_terms) if kw_terms else "не выделены"

        topic_id = int(np.argmax(self.vectorize_topic(text)))
        terms = self.topic_terms(topic_id, k=5)
        topic_terms_str = ", ".join(w for w, _ in terms)

        extraction = extract_relations(text)

        associations = []
        for word in kw_terms[:2]:
            sims = self.similar_words(word, k=2)
            associations.extend([w for w, _ in sims])

        assoc_str = ", ".join(
            dict.fromkeys(associations)
        ) or "нейтральный контекст"

        phrases = ", ".join(dict.fromkeys(extraction.key_phrases[:3]))

        similar_docs = self.similar_documents(
            text, k=1, preview_length=preview_length
        )
        similar = similar_docs[0][0] if similar_docs else ""

        slots = {
            "topic": f"#{topic_id + 1}",
            "topic_terms": topic_terms_str,
            "keywords": kw_str,
            "sentiment_class": sent_class,
            "confidence": f"{confidence * 100:.1f}",
            "relation": extraction.relation_summary(),
            "associations": assoc_str,
            "phrases": phrases,
            "similar": similar,
            "facts": facts_summary(extraction.facts),
        }

        return {
            "extraction": extraction,
            "commands": commands,
            "slots": slots,
            "sentiment": (sent_class, probs),
            "topic_id": topic_id,
            "topic_terms": topic_terms_str,
            "keywords": tfidf_top,
        }

    def generate(self, text: str, *, preview_length: int | None = None) -> str:
        summary = self.extract_semantic_summary(
            text, preview_length=preview_length
        )
        return build_text(
            summary["commands"],
            summary["slots"],
            summary["slots"]["sentiment_class"],
        )

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return joblib.load(Path(path))
