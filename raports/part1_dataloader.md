## Przygotowanie danych i pipeline uczący

### Źródło danych i automatyczne pobieranie

W eksperymentach wykorzystano zbiór danych **Medical Segmentation Decathlon – Task 05 (Prostate)**, obejmujący trójwymiarowe obrazy rezonansu magnetycznego prostaty wraz z odpowiadającymi im maskami segmentacyjnymi. Dane udostępniane są w formacie NIfTI oraz posiadają ustandaryzowaną strukturę katalogów opisaną w pliku `dataset.json`.

Proces przygotowania danych został w pełni zautomatyzowany. W przypadku braku lokalnej kopii zbioru danych archiwum jest pobierane z oficjalnego repozytorium Medical Segmentation Decathlon, a następnie rozpakowywane do wskazanego katalogu roboczego. Po zakończeniu ekstrakcji sprawdzana jest obecność pliku `dataset.json`, który zawiera informacje o podziale danych na zbiór treningowy oraz testowy. Takie rozwiązanie zapewnia powtarzalność eksperymentów oraz eliminuje konieczność ręcznej ingerencji użytkownika w strukturę danych.

Na podstawie zawartości pliku `dataset.json` tworzona jest lista próbek treningowych w postaci par *(obraz, maska)* oraz lista próbek testowych zawierających wyłącznie obrazy wejściowe. Ścieżki względne do plików są automatycznie konwertowane na ścieżki bezwzględne.

---

### Podział danych na zbiór treningowy i walidacyjny

Zbiór treningowy został losowo podzielony na część treningową oraz walidacyjną. Proporcja danych walidacyjnych wynosi 20% całkowitej liczby próbek treningowych. Podział realizowany jest z wykorzystaniem deterministycznego generatora liczb losowych z ustalonym ziarnem, co gwarantuje pełną odtwarzalność eksperymentów.

Losowanie odbywa się na poziomie całych wolumenów, bez naruszania spójności danych obraz–maska. Zbiór walidacyjny nie podlega augmentacji danych i służy wyłącznie do monitorowania procesu uczenia oraz doboru hiperparametrów.

---

### Standaryzacja danych wejściowych

Przed podaniem danych do sieci neuronowej wszystkie obrazy oraz maski segmentacyjne poddawane są zestandaryzowanemu procesowi wstępnego przetwarzania. Pipeline standaryzacji obejmuje następujące etapy:

1. **Wczytanie danych z dysku**  
   Obrazy MRI oraz odpowiadające im maski segmentacyjne są ładowane do pamięci operacyjnej na podstawie listy plików wygenerowanej z `dataset.json`.

2. **Ujednolicenie wymiarów kanałów**  
   Dane są konwertowane do formatu kanałowego *(C, H, W, D)*, zgodnego z konwencją stosowaną w bibliotece PyTorch, nawet jeśli obraz wejściowy posiada pojedynczy kanał intensywności.

3. **Ujednolicenie orientacji anatomicznej**  
   Wszystkie wolumeny są przekształcane do wspólnego układu współrzędnych RAS (Right–Anterior–Superior), co eliminuje różnice wynikające z odmiennej orientacji skanów MRI.

4. **Resampling przestrzenny (Spacing)**  
   Obrazy i maski są interpolowane do jednolitej rozdzielczości przestrzennej:
   - obraz MRI: interpolacja liniowa (bilinear),
   - maska segmentacyjna: interpolacja najbliższego sąsiada (nearest).

   Zapewnia to spójność geometryczną danych pochodzących z różnych badań.

5. **Normalizacja intensywności sygnału**  
   Intensywności obrazu MRI są skalowane liniowo z zakresu \([0, 3000]\) do przedziału \([0, 1]\). Wartości spoza tego zakresu są obcinane. Operacja ta stabilizuje proces uczenia i zmniejsza wrażliwość modelu na zmiany kontrastu.

6. **Automatyczne przycięcie obszaru zainteresowania**  
   Wolumeny są kadrowane do minimalnego obszaru obejmującego niezerowe wartości intensywności obrazu. Pozwala to ograniczyć ilość tła i zmniejszyć zapotrzebowanie na pamięć obliczeniową.

7. **Dopełnienie przestrzenne (padding)**  
   Obrazy i maski są dopełniane do zadanego rozmiaru przestrzennego ROI, co umożliwia dalsze przetwarzanie danych o różnych wymiarach wejściowych.

---

### Ekstrakcja fragmentów wolumetrycznych (ROI)

Ze względu na wysoką rozdzielczość danych 3D oraz ograniczenia pamięci GPU, uczenie modelu odbywa się na fragmentach wolumenów o stałym rozmiarze (*Region of Interest*).  

Dla zbioru treningowego stosowane jest losowe wycinanie fragmentów ROI z kontrolą proporcji obszarów zawierających strukturę docelową (prostatę) oraz tło. Mechanizm ten wymusza częstsze próbkowanie regionów zawierających etykietę, co przeciwdziała silnej nierównowadze klas typowej dla segmentacji medycznej.

Dla zbioru walidacyjnego i testowego nie stosuje się losowego próbkowania ROI – dane są jedynie dopełniane przestrzennie i przetwarzane w sposób deterministyczny.

---

### Augmentacja danych treningowych

W celu zwiększenia różnorodności danych treningowych oraz poprawy zdolności generalizacji modelu zastosowano zestaw losowych transformacji augmentacyjnych, wykonywanych wyłącznie na zbiorze treningowym. Augmentacje obejmują:

#### Augmentacje geometryczne
- losowe odbicia lustrzane względem osi X, Y oraz Z,
- losowe transformacje afiniczne obejmujące:
  - rotacje do ±15° wokół każdej osi,
  - skalowanie geometryczne,
  - translacje przestrzenne,
- interpolacja liniowa dla obrazu oraz interpolacja najbliższego sąsiada dla maski segmentacyjnej.

#### Augmentacje intensywnościowe
- losowe przesunięcie intensywności sygnału,
- losowe skalowanie intensywności obrazu.

#### Augmentacje szumowe i filtracyjne
- dodanie losowego szumu gaussowskiego o zerowej wartości średniej i niewielkim odchyleniu standardowym,
- losowe wygładzanie gaussowskie z losowo dobieranym parametrem sigma w każdej osi przestrzennej.

Zastosowanie powyższych transformacji pozwala symulować zmienność warunków akwizycji MRI, różnice pomiędzy aparatami oraz naturalne deformacje anatomiczne, zwiększając odporność modelu na zakłócenia i poprawiając jego zdolność uogólniania.

---

### Buforowanie danych i integracja z procesem uczenia

Pipeline danych wykorzystuje mechanizm buforowania `CacheDataset`, który umożliwia przechowywanie w pamięci przetworzonych próbek po etapie standaryzacji. Dzięki temu kosztowne operacje wczytywania i przetwarzania danych są wykonywane jednorazowo, co znacząco skraca czas kolejnych epok uczenia.

Całość pipeline’u została opakowana w moduł dziedziczący po `LightningDataModule`, co umożliwia bezpośrednią integrację z frameworkiem PyTorch Lightning oraz jednoznaczny podział danych na fazy treningu, walidacji i testów.
