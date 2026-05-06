# Projekt proponowanego rozwiązania

## Opis zbioru danych

### HDFS (Hadoop Distributed File System)

Zbiór HDFS pochodzi z benchmarku LogHub (Xu et al., 2009) i zawiera logi generowane przez rozproszony klaster Hadoop działający w środowisku akademickim. Każda linia logu reprezentuje zdarzenie związane z operacją na bloku danych (ang. *block*) i ma następujący format:

```
<DD MM YY> <HH MM SS> <thread_id> <LEVEL> <Component>: <message>
```

Przykład:
```
081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 ...
```

| Cecha | Wartość |
|---|---|
| Całkowita liczba linii logów | ~11,2 miliona |
| Liczba unikalnych bloków (block_id) | ~576 000 |
| Liczba anomalnych bloków | ~16 838 (~2,9%) |
| Jednostka anomalii | blok (block_id) |
| Etykiety | zewnętrzny plik `anomaly_label.csv` |

Etykiety anomalii są dostarczone jako oddzielny plik CSV mapujący `block_id → Normal / Anomalous`. Anomalia jest definiowana na poziomie całej sekwencji zdarzeń przypisanych do danego bloku: sekwencja jest anomalna, jeśli cykl życia bloku (alokacja → zapis → replikacja → zakończenie) został przerwany lub zawiera niezgodności.

### BGL (Blue Gene/L)

Zbiór BGL pochodzi z benchmarku LogHub i zawiera logi systemu zarządzania błędami sprzętowymi (ang. *Reliability, Availability and Serviceability* – RAS) superkomputera IBM Blue Gene/L zainstalowanego w Argonne National Laboratory. Format linii jest następujący:

```
<LABEL> <UNIX_TS> <DATE> <NODE_ID> <DATETIME> <NODE_ID> <COMPONENT> <SUBCOMPONENT> <LEVEL> <MESSAGE>
```

Przykład linii anomalnej:
```
KERNDTLB 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL FATAL instruction cache parity error corrected
```

| Cecha | Wartość |
|---|---|
| Całkowita liczba linii logów | ~4,63 miliona |
| Odsetek linii anomalnych | ~7,2% (ok. 332 222 linii) |
| Jednostka anomalii | linia logu (etykieta inline) |
| Grupowanie sekwencji | okna czasowe (sliding window) |
| Etykiety | inline — pierwszy token linii: `-` = normalny, kod alertu = anomalia |

W przeciwieństwie do HDFS, w zbiorze BGL etykiety anomalii są **wbudowane inline** w każdą linię logu. Pole `<LABEL>` przyjmuje wartość `-` dla zdarzeń normalnych lub kod alertu (np. `KERNDTLB`, `APPSEV`, `APPREAD`) dla zdarzeń anomalnych. Brak odrębnego identyfikatora sesji (odpowiednika `block_id`) powoduje, że sekwencjonowanie opiera się na oknach czasowych.

---

## Opis całego łańcucha przetwarzania

Proponowany system składa się z sześciu głównych etapów transformacji danych, prowadzących od surowych plików logów do wyuczonego modelu wykrywania anomalii. Poniższy diagram przedstawia ogólny przepływ danych:

```
Surowe logi (.log)
       │
       ▼
[1. Parsowanie + Template Mining]  ──── Drain3, masking
       │
       ▼ *_templates.json, *_annotated.parquet
       │
       ▼
[2. Wzbogacenie templateów z LLM]  ──── Azure AI / Mistral
       │
       ▼ *_templates_enriched.json
       │
       ▼
[3. Sekwencjonowanie]  ──── groupby block_id / sliding window
       │
       ▼ *_sequences.parquet
       │
       ▼
[4. Obliczenie embeddingów]  ──── TF-IDF + SBERT (all-MiniLM-L6-v2)
       │
       ▼
[5. Konstrukcja grafów]  ──── collapsed-template + pozycje + delty czasu
       │
       ▼ *_graph_dataset.pt (PyG Data)
       │
       ▼
[6. Trening autoenkodera (GAE)]  ──── GINEConv, multi-task loss
       │
       ▼ Model anomaly score per graph
```

---

## 1. Ingress logów

Surowymi danymi wejściowymi są pliki tekstowe `.log` przechowywane na dysku lokalnym bądź w zasobach chmurowych (ścieżki zarządzane przez DVC). Wersjonowanie danych odbywa się za pomocą narzędzia **DVC** (Data Version Control), które śledzi artefakty w pliku `dvc.yaml` i zapewnia reprodukowalność eksperymentów.

```
data/
  raw/
    HDFS_full.log        ─ 11,2 M linii logów HDFS
    BGL_full.log         ─ 4,63 M linii logów BGL
    anomaly_label.csv    ─ etykiety bloków HDFS
```

Pliki logów są odczytywane strumieniowo linia po linii (`open(..., errors="replace")`), co umożliwia przetwarzanie zbiorów przekraczających dostępną pamięć RAM.

---

## 2. Parsowanie i Template Mining (Drain3)

### Motywacja

Surowe wiersze logów zawierają zarówno statyczne fragmenty tekstowe (szablony, ang. *templates*), jak i zmienne parametry (adresy IP, identyfikatory bloków, wartości numeryczne). Celem etapu parsowania jest automatyczne odkrycie powtarzających się wzorców — tzw. **szablonów Drain** — oraz ekstrakcja zmiennych parametrów z każdej linii.

### Algorytm Drain (Drain3)

Drain (He et al., 2017) jest strumieniowym algorytmem grupowania logów (ang. *log template mining*) opartym na drzewie prefiksowym. Jego złożoność obliczeniowa jest liniowa względem liczby przetwarzanych linii dzięki ograniczonemu przeszukiwaniu drzewa.

**Struktura drzewa Drain:**

```
Korzeń (Root)
  └── węzeł głębokości 1: długość linii (po tokenizacji)
        └── węzeł głębokości 2..D: pierwsze D tokenów linii
              └── liść: klaster szablonu (LogCluster)
                    • template: bieżący szablon z <*> w miejscach zmiennych
                    • cluster_id: stabilny identyfikator klastra
```

**Pseudokod algorytmu Drain (online processing):**

```
FUNCTION drain_match(line, tree, sim_th, depth):
    tokens = tokenize(line)
    node = lookup_by_length(tree, len(tokens))
    
    FOR i IN range(min(depth, len(tokens))):
        IF tokens[i] is not wildcard:
            node = navigate(node, tokens[i])
    
    best_cluster = None
    best_sim = -1
    FOR cluster IN node.clusters:
        sim = seq_dist(tokens, cluster.template)
        IF sim > best_sim:
            best_sim = sim
            best_cluster = cluster
    
    IF best_sim >= sim_th:
        RETURN best_cluster        # aktualizacja szablonu (uogólnienie)
    ELSE:
        RETURN create_new_cluster(tokens)  # nowy szablon
    
FUNCTION seq_dist(tokens_a, tokens_b):
    equal = sum(1 FOR t_a, t_b IN zip(tokens_a, tokens_b) IF t_a == t_b)
    RETURN equal / len(tokens_a)
```

### Preprocessing – Masking

Przed przekazaniem linii do algorytmu Drain wykonywane jest **maskowanie** (ang. *masking*) zmiennych tokenów o wysokiej kardynalności. Operacja ta jest konfigurowana w plikach `configs/drain.ini` i `configs/drain_bgl.ini` za pomocą wyrażeń regularnych.

**HDFS – maski:**

| Wzorzec | Maska | Przykład |
|---|---|---|
| `blk_-?\d+` | `BLK` | `blk_-1608999687919862906` → `BLK` |
| `\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?` | `IP` | `10.1.2.3:50010` → `IP` |
| UUID (format 8-4-4-4-12) | `UUID` | |

**BGL – maski:**

| Wzorzec | Maska | Przykład |
|---|---|---|
| Rack/node IDs (`R\d+-M\d+-...`) | `NODE` | `R02-M1-N0-C:J12-U11` → `NODE` |
| IPv4 z opcjonalnym portem | `IP` | |
| Adresy szesnastkowe (`0x...`) | `HEX` | `0x00400000` → `HEX` |

### Konfiguracja parsera

Kluczowe hiperparametry Drain konfigurowane przez pliki `.ini`:

| Parametr | HDFS | BGL | Znaczenie |
|---|---|---|---|
| `sim_th` | 0.4 | 0.4 | Próg podobieństwa do przypisania do istniejącego klastra |
| `depth` | 4 | 4 | Głębokość drzewa prefiksowego |
| `max_children` | 100 | 100 | Maksymalny współczynnik rozgałęzienia węzła |
| `max_clusters` | 512 | 1024 | Górny limit liczby poznanych szablonów |

BGL posiada większy limit klastrów (1024 vs 512) ze względu na bogatsze słownictwo systemowe (~300–400 unikalnych szablonów na pełnym zbiorze LogHub).

### Dwuprzebiegowy schemat parsowania

Implementacja (`DrainParser`, `BGLParser`) stosuje **dwuprzebiegowe** przetwarzanie:

1. **Przebieg I – Dopasowanie (fit):** Algorytm Drain przetwarza wszystkie linie i uczy szablonów online. Szablony ewoluują — tokeny zmienne są zastępowane symbolem `<*>` w miarę napływu kolejnych linii. Na końcu tego etapu każdy klaster posiada ustabilizowany szablon.

2. **Przebieg II – Adnotacja (annotate):** Każda linia jest ponownie przetwarzana przy użyciu finalnych (w pełni uogólnionych) szablonów. Wynikiem jest ramka danych (DataFrame) z kolumnami:

**HDFS — kolumny wyjściowe:**

| Kolumna | Typ | Opis |
|---|---|---|
| `date`, `time`, `thread` | str | Surowe tokeny nagłówka |
| `timestamp` | datetime | Sparsowany czas zdarzenia |
| `raw` | str | Oryginalna linia logu |
| `cluster_id` | int | Stabilny identyfikator klastra Drain |
| `template` | str | Finalna postać szablonu (z `<*>`) |
| `parameters` | list | Wartości wyodrębnione dla każdego `<*>` |
| `block_id` | str / None | Pierwszy `blk_XXX` znaleziony w linii |

**BGL — kolumny dodatkowe:**

| Kolumna | Typ | Opis |
|---|---|---|
| `label` | str | Kod alertu (`-` = normalny) |
| `is_anomaly` | bool | `True` jeśli `label != "-"` |
| `unix_ts` | int | Uniksowy znacznik czasu |
| `node_id` | str | Identyfikator węzła obliczeniowego |
| `component`, `subcomponent`, `level` | str | Metadane zdarzenia RAS |

Wynik adnotacji jest zapisywany do pliku Parquet:

```python
# Przykład zapisu (encoder: fastparquet, ze względu na kompatybilność z Arrow-backed strings)
df_save["parameters"] = df_save["parameters"].apply(json.dumps)
df_save.to_parquet(PARQUET_PATH, index=False, engine="fastparquet")
```

### Walidacja szablonów

Po zakończeniu parsowania wywoływana jest metoda `parser.validate()`, która heurystycznie ocenia jakość nauczonych szablonów. Finalny zestaw szablonów jest eksportowany do pliku JSON:

```json
[
  {
    "cluster_id": 7,
    "template": "Receiving block BLK src: IP dest: IP",
    "example_lines": ["Receiving block blk_123 src: 10.0.0.1 ..."],
    "count": 4219
  },
  ...
]
```

---

## 3. Wzbogacenie szablonów logów z LLM

### Motywacja

Surowe szablony Drain są zwięzłe, ale pozbawione kontekstu semantycznego — szablon `"Receiving block BLK src: IP dest: IP"` sam w sobie nie niesie wiedzy o tym, czym jest `BLK`, jakie awarie mogą wystąpić w tym zdarzeniu ani co powinno następować po nim w normalnej sekwencji. Inspiracją dla tego kroku jest idea **Knowledge-Enriched Fusion** (Zhang et al., 2024, *"Log Anomaly Detection with Large Language Models via Knowledge-Enriched Fusion"*).

Celem wzbogacenia jest wygenerowanie dla każdego szablonu strukturalnego opisu semantycznego, który posłuży jako wejście do modelu embeddingowego SBERT, poprawiając jakość reprezentacji wektorowej szablonów.

### Schemat wzbogacenia (EnrichedTemplate)

Każdy szablon jest wzbogacany do ustrukturyzowanego obiektu `EnrichedTemplate` (zdefiniowanego za pomocą Pydantic):

| Pole | Typ | Opis |
|---|---|---|
| `component` | str | Komponent systemowy generujący log |
| `component_role` | str | Rola komponentu w systemie |
| `log_level` | str | Poziom logowania (INFO / WARN / ERROR) |
| `purpose` | str | Znaczenie zdarzenia w kontekście normalnej pracy |
| `fields` | `List[TemplateField]` | Semantyczny opis każdego pola `<*>` |
| `expected_sequence` | `List[str]` | Oczekiwana sekwencja szablonów przed i po |
| `failure_modes` | `List[FailureMode]` | Tryby awarii wykrywalne przez ten szablon |
| `anomaly_indicators` | `List[str]` | Konkretne sygnały anomalii |
| `related_templates` | `List[str]` | Powiązane szablony do korelacji w RCA |

Gdzie `TemplateField` opisuje każdy symbol `<*>` (nazwa, opis, relewantność dla anomalii), a `FailureMode` opisuje konkretny typ awarii (nazwa, opis, obserwowalny sygnał).

### Łańcuch wywołania LLM (LangChain)

Wzbogacenie realizowane jest poprzez łańcuch LangChain z modelem Mistral (Large i Small) udostępnionym przez Azure AI Foundry:

```python
# Pseudokod łańcucha
FUNCTION enrich_corpus_bgl(template):
    prompt = ChatPromptTemplate.from_messages([
        system_prompt(),
        human_message(context=BGL_CONTEXT, log_template=template)
    ])
    chain = prompt | llm.with_structured_output(EnrichedTemplate, method="json_mode")
    return chain.invoke({"log_template": template})
```

Parametry LLM:
- Model: `Mistral-Large-2407` (wzbogacenie główne) i `Mistral-Small-2409` (wzbogacenie porównawcze)
- `temperature=0` (deterministyczne wyjście)
- `method="json_mode"` — wymuszona struktura JSON (zamiast `json_schema`, który nie jest obsługiwany przez Azure AI Foundry)

**Kontekst dostarczony do modelu (HDFS):**

Prompt zawiera kontekst domenowy obejmujący:
- Opis architektury systemu (np. HDFS: NameNode + DataNode + cykl życia bloku)
- Definicję anomalii i trybów awarii (np. rozłączenie DataNode, awaria replikacji, uszkodzenie bloku)
- Reguły etykietowania (np. sekwencja `alokacja → zapis → replikacja → zakończenie`)

Wzbogacone szablony są eksportowane do JSON razem z surowymi szablonami:

```json
{
  "cluster_id": 7,
  "template": "Receiving block BLK src: IP dest: IP",
  "enriched_large": "{\"component\": \"DataNode$DataXceiver\", \"purpose\": \"...\", ...}",
  "enriched_small": "{...}"
}
```

---

## 4. Sekwencjonowanie

### HDFS — Sekwencjonowanie po identyfikatorze bloku (Session-based)

W zbiorze HDFS naturalna jednostka analizy to **blok HDFS** identyfikowany przez `block_id` (np. `blk_-1608999687919862906`). Każda operacja na bloku (alokacja, zapis, replikacja, usunięcie) jest rejestrowana osobną linią logu; wszystkie linie danego bloku tworzą jego sekwencję.

```python
# Sekwencjonowanie po block_id
df_blocks = df.loc[df["block_id"].notna()].sort_values(["block_id", "timestamp"])
sequences = dict(df_blocks.groupby("block_id", sort=False))
# Wynik: {block_id → DataFrame z liniami posortowanymi chronologicznie}
```

**Statystyki sekwencji HDFS:**
- Liczba unikalnych bloków: ~576 000
- Mediana długości sekwencji: ~20 zdarzeń
- Percentyl 99: ~29 zdarzeń (długie sekwencje są rzadkie)

### BGL — Sekwencjonowanie oknem czasowym (Sliding Time Window)

BGL nie posiada odpowiednika `block_id`. Sekwencje budowane są metodą **przesuwnego okna czasowego** (ang. *sliding time window*):

**Parametry bazowe:**
- Szerokość okna: `WINDOW_MINUTES = 5` minut
- Krok przesunięcia: `STEP_MINUTES = 1` minuta (50% nakładanie)

**Pseudokod algorytmu sliding window:**

```
FUNCTION build_sliding_windows(df, window_size, step_size):
    unix_arr = sorted array of unix timestamps
    t_min, t_max = unix_arr[0], unix_arr[-1]
    window_starts = [t_min, t_min+step, t_min+2*step, ..., t_max-window_size]
    
    result_rows = []
    FOR w_start IN window_starts:
        lo = searchsorted(unix_arr, w_start, side="left")
        hi = searchsorted(unix_arr, w_start + window_size, side="left")
        IF hi > lo:
            rows = df[lo:hi].copy()
            rows["window_id"] = w_start
            result_rows.append(rows)
    
    RETURN concatenate(result_rows)
    // Złożoność: O(W_count × log N), gdzie W_count = liczba okien, N = liczba linii
```

Okno jest etykietowane jako **anomalne**, jeśli co najmniej jedna z jego linii ma flagę `is_anomaly = True`.

**Statystyki sekwencji BGL:**
- Szacunkowa liczba okien: zależna od rozpiętości czasowej logu
- Zmienna długość sekwencji — wysoka wariancja (mediana ~46, maksimum ~177 000 zdarzeń)

---

## 5. Obliczenie embeddingów TF-IDF i SBERT

Każdy szablon Drain jest reprezentowany jako **hybrydowy wektor embeddingowy** będący konkatenacją dwóch komplementarnych reprezentacji:

### 5.1 TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF koduje statystyczne podobieństwo leksykalne szablonów. Korpusem jest zbiór wszystkich odkrytych szablonów Drain.

```python
# Przykład
tfidf_vec = TfidfVectorizer(analyzer="word", token_pattern=r"[^\s]+")
tfidf_matrix = tfidf_vec.fit_transform(all_templates)
tfidf_dense = tfidf_matrix.toarray()  # shape: (N_templates, N_vocab_terms)
```

Definicja wektora TF-IDF dla szablonu `d` i tokenu `t`:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)$$

gdzie:

$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t'} f_{t',d}}, \quad \text{IDF}(t) = \log\frac{1 + N}{1 + \text{df}(t)} + 1$$

- $f_{t,d}$ — liczba wystąpień tokenu $t$ w szablonie $d$
- $N$ — łączna liczba szablonów
- $\text{df}(t)$ — liczba szablonów zawierających token $t$

Wymiar: liczba unikalnych tokenów w szablonach (zależna od zbioru danych, typowo kilkaset do kilku tysięcy).

### 5.2 Sentence-BERT (SBERT, model `all-MiniLM-L6-v2`)

SBERT generuje **semantyczne embeddingi zdaniowe** z użyciem wytrenowanego transformera. Na wejście podawany jest tekst wzbogaconego szablonu (pole `purpose` + `failure_modes` z `EnrichedTemplate`). Jeśli wzbogacenie niedostępne — stosowany jest surowy tekst szablonu.

```python
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

enriched_texts = []
for cid in all_cids:
    info = cluster_to_enriched.get(cid)
    if info:
        parts = [info.get("purpose", "")]
        for fm in info.get("failure_modes", []):
            parts.append(f"{fm['name']}: {fm['description']}")
        text = " ".join(parts).strip()
    else:
        text = cluster_to_template.get(cid, "")
    enriched_texts.append(text or "unknown log template")

sbert_embeddings = sbert_model.encode(
    enriched_texts,
    show_progress_bar=True,
    normalize_embeddings=True   # L2-normalizacja → wektory jednostkowe
)
sbert_embeddings = np.nan_to_num(sbert_embeddings, nan=0.0)
```

Model `all-MiniLM-L6-v2`:
- Architektura: 6-warstwowy transformer (MiniLM), 22 M parametrów
- Wymiar embeddingu: **384**
- Wytrenowany na zbiorze MS MARCO + wielojęzycznych korpusach zdań
- Normalizacja L2 na wyjściu (wszystkie wektory leżą na hipersferze jednostkowej)

### 5.3 Hybrydowy wektor embeddingowy

Oba wektory są konkatenowane poziomo:

$$\mathbf{e}_{cid} = [\mathbf{v}^{\text{TF-IDF}}_{cid} \; \| \; \mathbf{v}^{\text{SBERT}}_{cid}] \in \mathbb{R}^{d_{\text{TF-IDF}} + 384}$$

Słownik embeddingów: `cluster_embeddings = {cluster_id → e_cid}` jest budowany raz i używany jako tablica przeglądowa podczas konstruowania grafów.

---

## 6. Konstrukcja grafów

### Motywacja i wybór reprezentacji

Każda sekwencja logów (blok HDFS lub okno BGL) jest przekształcana w **skierowany graf atrybutowany** (ang. *attributed directed graph*), stanowiący wejście dla grafowej sieci neuronowej (GNN).

Zbadano dwa podejścia do reprezentacji grafowej:

| | **A: Collapsed-template** | **B: One-node-per-event** |
|---|---|---|
| Tożsamość węzła | Jeden węzeł na unikalny `cluster_id` | Jeden węzeł na każdą linię logu |
| Rozmiar grafu | Mały (5–22 węzłów dla HDFS) | Duży (= długość sekwencji) |
| Powtarzające się zdarzenia | Atrybut `occurrence_count` | Oddzielne węzły |
| Krawędź (A→B) | Ważona, waga = liczba przejść | Nieważony łańcuch |
| Informacja temporalna | Zagregowana (rozkład delt czasu) | Dokładna (per krawędź) |
| Dopasowanie do GNN | Doskonałe (inwarianty strukturalne) | Zasadniczo lista łączona (zła dla GNN) |

**Wybrane rozwiązanie: podejście A — Collapsed-template z cechami pozycyjnymi.**

Grafy kolapsowane są lepszym wejściem dla GNN, ponieważ:
1. Sekwencje o podobnym wzorcu zdarzeń dają izomorficzne grafy — GNN może uczyć się *sygnatur strukturalnych* normalnych i anomalnych bloków.
2. Zamiast sekwencji liniowej (ograniczonej informacji dla GNN) otrzymujemy kompaktową reprezentację topologiczną.
3. Informacja o kolejności zdarzeń (tracona przy kolapsowaniu) jest w pełni odzyskiwalna poprzez **cechy pozycyjne** węzłów i krawędzi.

### Definicja struktury grafu

#### Wierzchołki (Nodes)

Każdy *unikalny* szablon Drain (`cluster_id`) występujący w sekwencji staje się jednym węzłem grafu.

**Wektor cech węzła** $\mathbf{x}_v \in \mathbb{R}^{d_{\text{emb}} + 9}$:

| Indeks | Cecha | Opis |
|---|---|---|
| `0 : d_emb` | Hybrydowy embedding | TF-IDF + SBERT (patrz §5) |
| `d_emb` | `occurrence_count` | Liczba wystąpień szablonu w sekwencji |
| `d_emb+1` | `param_count` | Łączna liczba wyodrębnionych parametrów |
| `d_emb+2` | `param_num_mean` | Średnia wartości numerycznych wśród parametrów |
| `d_emb+3` | `param_num_max` | Maksymalna wartość numeryczna wśród parametrów |
| `d_emb+4` | `first_pos` | Znormalizowana pozycja pierwszego wystąpienia ∈ [0,1] |
| `d_emb+5` | `last_pos` | Znormalizowana pozycja ostatniego wystąpienia ∈ [0,1] |
| `d_emb+6` | `mean_pos` | Średnia znormalizowana pozycja wszystkich wystąpień |
| `d_emb+7` | `std_pos` | Odchylenie standardowe pozycji (0 jeśli 1 wystąpienie) |
| `d_emb+8` | `pos_spread` | `last_pos − first_pos` |

Pozycja $i$-tego zdarzenia jest normalizowana: $p_i = i / \max(n-1, 1)$, gdzie $n$ = długość sekwencji.

**Uzasadnienie cech pozycyjnych:**

Cechy te umożliwiają GNN odróżnienie anomalii porządkowych. Przykład: sekwencja A→B→A→C vs A→A→B→C — oba dają ten sam graf kolapsowany bez cech pozycyjnych, ale z cechami:
- A→B→A→C: węzeł A ma `mean_pos=0.33`, `std_pos=0.47`
- A→A→B→C: węzeł A ma `mean_pos=0.17`, `std_pos=0.24`

Te wektory cech są **różne** — GNN może nauczyć się tej różnicy.

#### Krawędzie (Edges)

Krawędź skierowana $u \to v$ istnieje, jeśli w sekwencji szablon $u$ był natychmiast poprzedzony przez szablon $v$ co najmniej raz. Krawędź jest agregatem wszystkich przejść $u \to v$.

**Wektor cech krawędzi** $\mathbf{e}_{uv} \in \mathbb{R}^{10}$:

| Indeks | Cecha | Opis |
|---|---|---|
| `0` | `weight` | Liczba przejść $u \to v$ |
| `1` | `td_min` | Minimalna delta czasu [s] |
| `2` | `td_p25` | 25. percentyl delt czasu [s] |
| `3` | `td_median` | Mediana delt czasu [s] |
| `4` | `td_p75` | 75. percentyl delt czasu [s] |
| `5` | `td_max` | Maksymalna delta czasu [s] |
| `6` | `td_std` | Odchylenie standardowe delt czasu [s] |
| `7` | `mean_src_pos` | Średnia znorm. pozycja źródłowego węzła |
| `8` | `mean_dst_pos` | Średnia znorm. pozycja docelowego węzła |
| `9` | `mean_pos_delta` | Średnie `dst_pos − src_pos` |

7-wymiarowy rozkład delty czasu (`td_*`) pozwala GNN wykrywać zarówno powolne przejścia (np. timeout replikacji: wysoki percentyl `p75`) jak i brakujące przejścia (`weight = 0` w grafie kolapsowanym).

Jeśli znaczniki czasowe nie są dostępne, wszystkie cechy temporalne przyjmują wartość `-1` (sentinel).

**Pseudokod budowy grafu:**

```
FUNCTION build_collapsed_graph(sequence):
    // Agregacja na węzłach
    node_count = Counter(sequence.cluster_ids)
    node_positions = defaultdict(list)     // cluster_id → [normalized positions]
    
    FOR i, (cid, params) IN enumerate(sequence):
        pos = i / max(len(sequence) - 1, 1)
        node_positions[cid].append(pos)
    
    // Agregacja na krawędziach
    edge_deltas    = defaultdict(list)
    edge_positions = defaultdict(list)
    
    FOR i IN range(len(sequence) - 1):
        src, dst = sequence[i].cluster_id, sequence[i+1].cluster_id
        edge_positions[(src, dst)].append((i/(n-1), (i+1)/(n-1)))
        IF timestamps available:
            edge_deltas[(src, dst)].append(ts[i+1] - ts[i])
    
    // Budowa grafu
    G = nx.DiGraph()
    FOR cid, positions IN node_positions.items():
        G.add_node(cid, **build_node_features(cid, positions, ...))
    FOR (src, dst), positions IN edge_positions.items():
        G.add_edge(src, dst, **build_edge_features(positions, edge_deltas[(src, dst)]))
    
    RETURN G
```

### Konwersja do formatu PyG (PyTorch Geometric)

Każdy graf NetworkX jest konwertowany do obiektu `torch_geometric.data.Data`:

```python
Data(
    x          = torch.FloatTensor([num_nodes, NODE_DIM]),   # macierz cech węzłów
    edge_index = torch.LongTensor([2, num_edges]),           # COO format sąsiedztwa
    edge_attr  = torch.FloatTensor([num_edges, EDGE_DIM]),   # macierz cech krawędzi
    y          = torch.LongTensor([1]),                      # etykieta (0=normalny, 1=anomalny)
    block_id   = str,                                        # identyfikator bloku/okna
    num_nodes  = int,
)
```

### Podział na zbiory (stratyfikowany)

Dataset jest dzielony stratyfikowaną metodą zgodnie z proporcją anomalii:

- **Zbiór treningowy (Train):** 70%
- **Zbiór walidacyjny (Val):** 15%
- **Zbiór testowy (Test):** 15%

Podział realizowany jest funkcją `train_test_split` z biblioteki `scikit-learn` z parametrem `stratify=labels` i `random_state=42`, gwarantując identyczną proporcję anomalii we wszystkich zbiorach.

---

## 7. Definicja i uczenie autoenkodera grafowego (GAE)

### Paradygmat wykrywania anomalii

System stosuje **nienadzorowane uczenie maszynowe** w paradygmacie **Clean-Train**:
- Model jest trenowany **wyłącznie na normalnych grafach** (etykieta `y=0`)
- Model uczy się rekonstruować normalne wzorce struktury logów
- W fazie inferencji grafy o wysokim błędzie rekonstrukcji są flagowane jako anomalie

*Uzasadnienie:* W realnych środowiskach produkcyjnych anomalie są rzadkie i często nieoznakowane. Trening wyłącznie na danych normalnych eliminuje potrzebę etykietowania danych treningowych.

### Architektura modelu: AttributeAwareGAE

Model implementuje **Attribute-Aware Graph Autoencoder** z wielozadaniowym dekoderem:

```
                    ┌─────────────────────────────────────────┐
                    │         AttributeAwareGAE               │
                    │                                         │
Input x ──BatchNorm─┤                                         │
Input e ──BatchNorm─┤                                         │
                    │  ┌──────────────────────────────────┐   │
                    │  │            ENCODER               │   │
                    │  │  Linear(node_dim → hidden_dim)   │   │
                    │  │  Linear(edge_dim → hidden_dim)   │   │
                    │  │  GINEConv(hidden → latent_dim)   │   │
                    │  └──────────────┬───────────────────┘   │
                    │                 │ Z ∈ R^(N × latent_dim)│
                    │    ┌────────────┼────────────┐          │
                    │    │            │            │          │
                    │  ┌─▼──┐    ┌───▼───┐   ┌────▼────┐    │
                    │  │Dec1│    │ Dec2  │   │  Dec3   │    │
                    │  │Dot │    │Linear │   │ Linear  │    │
                    │  │Prod│    │(→node)│   │(2z→edge)│    │
                    │  └─┬──┘    └───┬───┘   └────┬────┘    │
                    │    │           │             │         │
                    │ Adj logits  x_hat         ea_hat       │
                    └─────────────────────────────────────────┘
```

**Szczegółowy opis komponentów:**

#### Normalizacja wejściowa

```python
self.raw_node_norm = nn.BatchNorm1d(node_dim, affine=False)
self.raw_edge_norm = nn.BatchNorm1d(edge_dim, affine=False)
```

BatchNorm bez uczonych parametrów (`affine=False`) standaryzuje wejście do ~N(0,1), zapobiegając eksplozji gradientów wynikającej ze skali wektorów TF-IDF.

#### Enkoder (GINEConv)

Enkoder składa się z dwóch warstw projekcji i jednej warstwy grafowej:

```python
# Projekcje wejściowe
self.node_proj = nn.Linear(node_dim, hidden_dim)   # hidden_dim = 128
self.edge_proj = nn.Linear(edge_dim, hidden_dim)

# Warstwa GINEConv
mlp = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.BatchNorm1d(hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, latent_dim)              # latent_dim = 64
)
self.encoder_conv = GINEConv(mlp, edge_dim=hidden_dim)
```

**GINEConv** (Graph Isomorphism Network with Edge features, Hu et al., 2019) to warstwa grafowego przekazywania wiadomości (ang. *message passing*) uwzględniająca cechy krawędzi:

$$\mathbf{h}_v^{(k)} = \text{MLP}^{(k)}\left( (1 + \epsilon^{(k)}) \cdot \mathbf{h}_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} \text{ReLU}\left(\mathbf{h}_u^{(k-1)} + \mathbf{e}_{uv}^{(k-1)}\right) \right)$$

gdzie:
- $\mathbf{h}_v^{(k)}$ — reprezentacja węzła $v$ po $k$-tej iteracji
- $\mathcal{N}(v)$ — zbiór sąsiadów węzła $v$
- $\mathbf{e}_{uv}$ — wektor cech krawędzi $(u, v)$
- $\epsilon$ — uczony (lub stały) parametr

Wyjściem enkodera jest macierz latentna $Z \in \mathbb{R}^{N \times d_z}$ ($d_z = 64$).

#### Dekoder 1 — Struktura grafu (Inner Product)

Rekonstrukcja macierzy sąsiedztwa przez iloczyn skalarny wektorów latentnych:

$$\hat{A}_{uv} = \sigma\left(\mathbf{z}_u \cdot \mathbf{z}_v^T\right)$$

implementowana jako zwrócenie surowych logitów (BCEWithLogitsLoss zastosowany zewnętrznie).

#### Dekoder 2 — Cechy węzłów

```python
self.node_decoder = nn.Sequential(
    nn.Linear(latent_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, node_dim)
)
```

Rekonstruuje znormalizowany wektor cech węzła $\hat{\mathbf{x}}_v$ z $\mathbf{z}_v$.

#### Dekoder 3 — Atrybuty krawędzi

```python
self.edge_decoder = nn.Sequential(
    nn.Linear(latent_dim * 2, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, edge_dim)
)
```

Na wejście podaje się konkatenację wektorów latentnych węzłów końcowych krawędzi: $[\mathbf{z}_u \| \mathbf{z}_v]$.

### Wielozadaniowa funkcja straty

Funkcja straty jest ważoną sumą trzech komponentów:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{struct}} + \beta \cdot \mathcal{L}_{\text{node}} + \gamma \cdot \mathcal{L}_{\text{edge}}$$

**Parametry domyślne:** $\alpha = \beta = \gamma = 1.0$ (zbilansowane wagi — justyfikacja: BatchNorm po stronie wejścia wyrównuje skale strat)

#### L1 — Strata strukturalna (Binary Cross-Entropy z próbkowaniem negatywnym)

```
L_struct = BCE(pos_logits, ones) + BCE(neg_logits, zeros)
```

Dla każdego grafu dobierana jest losowa próba **negatywnych krawędzi** (krawędzi nieistniejących w grafie) o tej samej liczności co krawędzie pozytywne:

```python
neg_edge_index = negative_sampling(
    batch.edge_index,
    num_nodes=batch.num_nodes,
    num_neg_samples=batch.edge_index.size(1)
)
```

#### L2 — Strata rekonstrukcji cech węzłów (MSE)

$$\mathcal{L}_{\text{node}} = \frac{1}{|V|} \sum_{v \in V} \|\hat{\mathbf{x}}_v - \tilde{\mathbf{x}}_v\|^2$$

gdzie $\tilde{\mathbf{x}}_v$ to znormalizowane (po BatchNorm) cechy węzła.

#### L3 — Strata rekonstrukcji atrybutów krawędzi (MSE)

$$\mathcal{L}_{\text{edge}} = \frac{1}{|E|} \sum_{(u,v) \in E} \|\hat{\mathbf{e}}_{uv} - \tilde{\mathbf{e}}_{uv}\|^2$$

### Algorytm treningu

```
FUNCTION train(model, train_loader, EPOCHS):
    optimizer = Adam(model.params, lr=1e-3)
    
    FOR epoch IN range(EPOCHS):      // domyślnie 25 epok
        model.train()
        FOR batch IN train_loader:   // batch_size = 256 grafów
            z, x_norm, ea_norm = model(batch.x, batch.edge_index, batch.edge_attr)
            
            // Oblicz straty
            L_str  = BCE(pos_logits) + BCE(neg_logits)
            L_node = MSE(decode_nodes(z), x_norm)
            L_edge = MSE(decode_edges(z, edge_index), ea_norm)
            
            L = α·L_str + β·L_node + γ·L_edge
            
            L.backward()
            clip_grad_norm_(model.params, max_norm=1.0)  // zapobiega eksplozji gradientów
            optimizer.step()
```

Hiperparametry:
- `HIDDEN_DIM = 128`
- `LATENT_DIM = 64`
- `BATCH_SIZE = 256`
- `EPOCHS = 25`
- `LEARNING_RATE = 1e-3` (HDFS: 1e-2; BGL: 1e-3, niższe ze względu na wyższy wymiar TF-IDF)

### Ocena anomalii (Inference)

W fazie inferencji błąd rekonstrukcji jest obliczany dla każdego grafu osobno i agregowany do **wyniku anomalii** (ang. *anomaly score*):

$$s(G) = \alpha \cdot \mathcal{E}_{\text{struct}}(G) + \beta \cdot \mathcal{E}_{\text{node}}(G) + \gamma \cdot \mathcal{E}_{\text{edge}}(G)$$

Agregacja odbywa się na poziomie grafu przy użyciu funkcji `scatter(reduce='mean', dim_size=num_graphs)`.

### Wyznaczenie progu decyzyjnego

Optymalny próg klasyfikacji jest wyznaczany na **zbiorze walidacyjnym** metodą krzywej Precision-Recall:

```python
precision, recall, thresholds = precision_recall_curve(val_labels, val_scores)
f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]
```

### Metryki ewaluacyjne

Do oceny końcowej na zbiorze testowym stosowane są:

- **ROC-AUC** — pole pod krzywą ROC (miara odporna na niezbalansowanie klas)
- **PR-AUC** — pole pod krzywą Precision-Recall
- **F1-Score** — harmoniczna średnia precyzji i kompletności (przy optymalnym progu)
- **Macierz pomyłek** — TP, FP, TN, FN
- **Classification Report** — precyzja, kompletność, F1 per klasie

---

## Opis wykorzystanych bibliotek zewnętrznych

### Przetwarzanie logów

#### drain3 (≥ 0.9)
Biblioteka implementująca strumieniowy algorytm Drain do wykrywania szablonów logów. Dostarcza:
- `TemplateMiner` — główny procesor online z wewnętrznym drzewem Drain
- `TemplateMinerConfig` / `FilePersistence` — konfiguracja i persystencja stanu parsera
- Metoda `add_log_message(line)` — przetwarza linię i zwraca `{cluster_id, template, params}`

#### pandas (≥ 2.2.0)
Biblioteka do analizy i transformacji danych tabelarycznych. Kluczowe zastosowania:
- `DataFrame` — przechowywanie adnotowanych logów (`*_annotated.parquet`)
- `groupby` — grupowanie linii po `block_id` i `window_id`
- Wejście/wyjście Parquet przez `to_parquet` / `read_parquet` z silnikiem `fastparquet`

#### numpy (≥ 1.26.0)
Biblioteka obliczeń numerycznych. Zastosowania:
- Operacje na tablicach dla sliding window (`np.searchsorted`, `np.arange`, `np.concatenate`)
- Obliczenia statystyk rozkładu delt czasu (`np.percentile`, `np.median`, `np.std`)
- Budowa macierzy embeddingów (`np.hstack`, `np.zeros`, `np.nan_to_num`)

### Modelowanie językowe i wzbogacanie

#### langchain-core / langchain-openai (≥ 0.2)
Framework do budowania łańcuchów wywołań LLM. Zastosowania:
- `ChatPromptTemplate` — strukturyzowanie promptów systemowych i użytkownika
- `ChatOpenAI` — klient API Azure AI Foundry (kompatybilny z OpenAI API)
- `llm.with_structured_output(EnrichedTemplate, method="json_mode")` — wymuszanie ustrukturyzowanego JSON na wyjściu modelu

#### pydantic (≥ 2.0)
Biblioteka walidacji danych i serializacji opartej na typach Pythona. Zastosowania:
- `BaseModel` — definicja schematu `EnrichedTemplate`, `TemplateField`, `FailureMode`
- Automatyczna walidacja struktury odpowiedzi LLM
- `model_dump_json()` — serializacja do JSON

#### sentence-transformers (≥ 2.6.0)
Biblioteka do generowania semantycznych embeddingów zdaniowych. Zastosowania:
- `SentenceTransformer("all-MiniLM-L6-v2")` — ładowanie wytrenowanego modelu
- `model.encode(texts, normalize_embeddings=True)` — batch encoding z L2-normalizacją
- Transformacja wzbogaconych opisów szablonów na wektory 384-wymiarowe

### Uczenie maszynowe i GNN

#### torch (PyTorch, ≥ 2.2.0)
Podstawowy framework głębokiego uczenia. Zastosowania:
- `nn.Module`, `nn.Linear`, `nn.BatchNorm1d`, `nn.Sequential`, `nn.ReLU` — warstwy modelu
- `F.mse_loss`, `F.binary_cross_entropy_with_logits` — funkcje straty
- `torch.optim.Adam` — optymalizator
- `torch.nn.utils.clip_grad_norm_` — obcinanie gradientów
- Tensor operacje GPU/MPS/CPU przez `torch.device`

#### torch-geometric (PyG, ≥ 2.5.0)
Biblioteka GNN oparta na PyTorch. Kluczowe komponenty:

- **`Data`** (`torch_geometric.data.Data`) — kontener danych grafu w formacie COO: `x`, `edge_index`, `edge_attr`, `y`
- **`DataLoader`** (`torch_geometric.loader.DataLoader`) — mini-batch'owanie grafów zmiennej wielkości przez automatyczne łączenie (ang. *batching via block diagonal adjacency*), atrybut `batch` mapuje węzły do grafów
- **`GINEConv`** (`torch_geometric.nn.GINEConv`) — warstwa message passing Graph Isomorphism Network with Edge features; uwzględnia cechy krawędzi w agregacji sąsiadów
- **`negative_sampling`** (`torch_geometric.utils.negative_sampling`) — losowe próbkowanie nieistniejących krawędzi dla treningu strukturalnego dekodera
- **`scatter`** (`torch_geometric.utils.scatter`) — agregacja wartości per-węzeł/krawędź na poziom grafu (`reduce='mean'`) z obsługą mini-batch

#### scikit-learn (≥ 1.4.0)
Biblioteka klasycznego uczenia maszynowego. Zastosowania:
- `TfidfVectorizer` — budowa macierzy TF-IDF dla szablonów Drain; parametry: `analyzer="word"`, `token_pattern=r"[^\s]+"` (traktuje każdy token nie-whitespace jako termin)
- `train_test_split` — stratyfikowany podział na Train/Val/Test z `stratify=labels`
- `precision_recall_curve`, `roc_auc_score`, `auc`, `f1_score`, `confusion_matrix`, `classification_report` — metryki ewaluacyjne

### Wizualizacja i analiza

#### matplotlib (≥ 3.8.0)
Biblioteka wizualizacji danych. Zastosowania:
- Histogramy rozkładu długości sekwencji
- Wykresy krzywych uczenia (strata w zależności od epoki)
- Wizualizacja grafów bloków

#### seaborn (≥ 0.13.0)
Biblioteka wizualizacji statystycznej oparta na matplotlib. Zastosowania:
- Heatmapy macierzy pomyłek (`sns.heatmap`)
- Zaawansowane wizualizacje rozkładów

#### networkx (≥ 3.2)
Biblioteka do tworzenia, analizy i wizualizacji grafów. Zastosowania:
- `nx.DiGraph` — budowa skierowanego grafu kolapsowanego z atrybutami węzłów i krawędzi w etapie eksploracyjnym (notebook `4_Logs2Graphs.ipynb`)
- `nx.spring_layout` — układ grafów do wizualizacji
- Etap pośredni: grafy NetworkX są konwertowane do formatu PyG

### MLOps i infrastruktura

#### DVC (Data Version Control, ≥ 3.50)
Narzędzie do wersjonowania danych i zarządzania potokami ML. Zastosowania:
- Śledzenie dużych plików danych (`.dvc` stubs w repozytorium Git)
- Definiowanie i uruchamianie kroków pipelinu w `dvc.yaml`
- Reprodukowalność eksperymentów

#### python-dotenv
Wczytywanie zmiennych środowiskowych z pliku `.env` (klucze API Azure, endpointy).

#### fastparquet
Alternatywny silnik Parquet dla pandas, używany zamiast `pyarrow` ze względu na inkompatybilności z kolumnami `Arrow-backed string` w niektórych konfiguracjach.

#### tqdm (≥ 4.66)
Paski postępu dla pętli (wzbogacanie szablonów, budowa grafów PyG, trening).

---

## Podsumowanie architektury

Poniżej zestawiono wymiary kluczowych tensorów w modelu (dla zbioru HDFS):

| Tensor | Wymiar | Opis |
|---|---|---|
| `x` (cechy węzłów) | `(N, d_TF-IDF + 384 + 9)` | Embedding hybrydowy + cechy pozycyjne + licznikowe |
| `edge_attr` (cechy krawędzi) | `(E, 10)` | Waga + 7 statystyk delt czasu + 3 cechy pozycyjne |
| `Z` (przestrzeń latentna) | `(N, 64)` | Reprezentacja latentna węzłów po GINEConv |
| Wynik anomalii `s(G)` | `(1,)` | Skalarna miara anomalności grafu |

System realizuje w pełni nienadzorowane wykrywanie anomalii, ucząc się normalnych wzorców struktury grafów logów i flagując sekwencje o wysokim błędzie rekonstrukcji jako potencjalne anomalie.
