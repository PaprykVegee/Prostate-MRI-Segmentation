## 3. Modele

W niniejszej pracy analizowane są modele głębokiego uczenia z rodziny **VNet**, przeznaczone do segmentacji wolumetrycznej obrazów medycznych. Architektury te wywodzą się bezpośrednio z koncepcji **U-Net**, która pierwotnie została zaprojektowana do segmentacji obrazów dwuwymiarowych. VNet stanowi jej naturalne rozszerzenie na dane trójwymiarowe, umożliwiając jednoczesne modelowanie zależności przestrzennych w osiach wysokości, szerokości oraz głębokości. Takie podejście jest szczególnie istotne w przypadku danych MRI, gdzie struktury anatomiczne wykazują ciągłość pomiędzy kolejnymi przekrojami.

Podobnie jak U-Net, architektury z rodziny VNet opierają się na symetrycznej strukturze typu *encoder–decoder* z połączeniami typu *skip connection*. Część kodująca odpowiada za ekstrakcję cech o rosnącym poziomie abstrakcji, natomiast część dekodująca rekonstruuje mapę segmentacji w oryginalnej rozdzielczości przestrzennej. Połączenia pomiędzy odpowiadającymi sobie poziomami enkodera i dekodera pozwalają na efektywne łączenie informacji semantycznej z dokładną lokalizacją przestrzenną obiektów.

---

### 3.1 VNet

Klasyczny VNet składa się z trójwymiarowego enkodera, dekodera oraz warstwy wyjściowej generującej mapy przynależności klas. Encoder realizuje proces stopniowej redukcji rozdzielczości poprzez konwolucje 3D z krokiem większym od jedności, jednocześnie zwiększając liczbę kanałów cech. Każdy poziom enkodera zawiera bloki rezydualne, które umożliwiają skuteczniejsze uczenie głębokiej sieci oraz redukują ryzyko zanikania gradientów. Zastosowanie funkcji aktywacji PReLU pozwala na lepsze dopasowanie nieliniowości do danych medycznych, w których rozkład intensywności może być silnie zróżnicowany.

Decoder VNetu realizuje proces odwrotny, polegający na stopniowym zwiększaniu rozdzielczości przestrzennej map cech. W tym celu wykorzystywane są transponowane konwolucje 3D, które umożliwiają jednoczesny upsampling i uczenie parametrów rekonstrukcji. Na każdym poziomie dekodera do aktualnych map cech dodawane są informacje pochodzące z odpowiadających poziomów enkodera poprzez połączenia typu skip connection. Dzięki temu możliwe jest zachowanie precyzyjnych informacji lokalizacyjnych, które zostały częściowo utracone w trakcie procesu downsamplingu.

Ostateczna warstwa konwolucyjna o rozmiarze jądra 1×1×1 mapuje cechy dekodera na przestrzeń klas segmentacji, generując dla każdego voxela wektory logitów, które następnie mogą zostać przekształcone w prawdopodobieństwa klasowe.

<div style="text-align: center;">
  <img src="raports/vnet.png" alt="Opis obrazka">
</div>

---

### 3.2 Attention VNet

Attention VNet stanowi rozwinięcie klasycznego VNetu poprzez wprowadzenie mechanizmu atencji w połączeniach typu skip connection. Głównym celem tego rozszerzenia jest umożliwienie sieci selektywnego skupienia się na najbardziej istotnych obszarach obrazu, zamiast bezwarunkowego przekazywania wszystkich cech z enkodera do dekodera. Jest to szczególnie ważne w segmentacji medycznej, gdzie struktury docelowe często zajmują niewielką część obrazu, a tło może dominować objętościowo.

Mechanizm atencji (*Attention Gate*) filtruje cechy przekazywane z enkodera przed ich integracją w dekoderze. Składa się z dwóch równoległych transformacji cech: **W_g** oraz **W_x**. Transformacja **W_g** działa na sygnale pochodzącym z głębszej warstwy dekodera, który zawiera informacje semantyczne o obecności struktury, lecz charakteryzuje się niską rozdzielczością przestrzenną. Transformacja **W_x** przetwarza cechy z odpowiadającej warstwy enkodera, które mają wysoką rozdzielczość przestrzenną i zawierają szczegółowe informacje lokalizacyjne, ale same w sobie nie dostarczają pewności co do istotności danego regionu.

Oba sygnały są łączone poprzez dodanie i poddawane funkcji aktywacji ReLU, która selekcjonuje jedynie te regiony, w których zarówno sygnał głęboki, jak i sygnał lokalny wskazują potencjalną obecność obiektu. Następnie wynik jest przekształcany w mapę wag przy użyciu funkcji sigmoidalnej, co pozwala na skalowanie wartości cech z enkodera w zakresie od 0 do 1. W efekcie tylko najbardziej istotne regiony są wzmocnione i przekazywane do dekodera, podczas gdy cechy nieistotne lub zakłócające są tłumione. Taki mechanizm pozwala sieci koncentrować się na kluczowych strukturach anatomicznych, zwiększając precyzję segmentacji i redukując liczbę fałszywie pozytywnych predykcji.

W praktyce Attention VNet zachowuje strukturę encoder–decoder klasycznego VNetu, przy czym każda warstwa dekodera otrzymuje przefiltrowane cechy z odpowiadającej warstwy enkodera. Dzięki temu model lepiej wykorzystuje kontekst semantyczny z głębszych warstw oraz szczegółowe cechy lokalne z warstw powierzchownych, co przekłada się na wyższą jakość segmentacji w porównaniu do klasycznego VNetu.

---

### 3.3 Różnice pomiędzy VNet i Attention VNet

Podstawowa różnica pomiędzy klasycznym VNetem a Attention VNetem dotyczy sposobu integracji informacji z połączeń typu skip connection. W VNecie informacje te są przekazywane w sposób bezpośredni i nieprzefiltrowany, co oznacza, że decoder otrzymuje pełny zestaw cech niezależnie od ich znaczenia dla aktualnego zadania segmentacji. Choć podejście to jest skuteczne w wielu przypadkach, może prowadzić do propagowania cech tła oraz zwiększenia liczby fałszywie pozytywnych predykcji.

Attention VNet wprowadza mechanizm selektywnego przekazywania informacji, który umożliwia sieci dynamiczne ważenie cech w zależności od kontekstu semantycznego. Dzięki wykorzystaniu sygnału z głębszych warstw dekodera, bramki atencji są w stanie wskazać regiony, w których z dużym prawdopodobieństwem występuje poszukiwana struktura. W efekcie model lepiej radzi sobie z segmentacją małych, słabo widocznych lub niejednoznacznych obiektów, a granice segmentacji są bardziej precyzyjne.

Zastosowanie mechanizmu atencji wiąże się z niewielkim wzrostem liczby parametrów oraz złożoności obliczeniowej modelu, jednak w praktyce koszt ten jest kompensowany przez poprawę jakości segmentacji oraz większą stabilność predykcji. Attention VNet stanowi zatem bardziej zaawansowaną i adaptacyjną wersję klasycznego VNetu, szczególnie dobrze przystosowaną do złożonych danych medycznych 3D.
