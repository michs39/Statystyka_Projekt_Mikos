# Projekt – Naiwny Klasyfikator Bayesowski w języku Python

Michał Mikoś\

# Wstępna analiza danych

Mushroom:\
Ilość grzybów jadalnych jest bliska ilości grzybów trujących\
veil-type jest takie samo dla wszytkich danych - można usunąć kolumnę\
![korelacja grzybow](/mushroom_corr.png)
jak widać na załączonym obrazku pojedyncze kolumny które mają duży wpływ na rodzaj grzyba to:\
bruises,\
gill-size,\
gill-color,\
ring-type\

Iris:\
Tyle samo każdego rodzaju kwiatów\
![korelacja kwiatów](/iris_corr.png)
Jak widać, z jednej strony petal-width i petal-length korelują z innymi danymi w "iris" co nie jest zgodne z założeniami metody i może negatywnie wpłynąć na wyniki, natomiast z drugiej strony mają duży wpływ na rodzaj kwiatów co moze wpłynąć na nie pozytywnie\

# Opis metody:

Metoda jest oparta o założenia twierdzenia Bayesa i niezależności zdarzeń.\
Składa się z trzech części:\
fit\
Wytrenowanie modelu, podzielenie danych na części odpowiadające każdemu rodzajowi grzyba/kwiatu i policzenie ilości/średniej i odchylenia standardowego\
predict_proba\
Na podstawie policzonych danych policzyć liczby odpowiadające prawdopodobieństwu dla każdego rodzaju grzyba/kwiatu\
predict\
Znalezienie największej liczby wyliczonej przez predict_proba i zwrócenie odpowiadajacego jej rodzaju\

# Wyniki

Dokładność modelu dla zmiennych kategorycznych - ok. 99%\
Dokładność modelu dla zmiennych ciągłych - ok. 95%

# Wykorzystane materiały:

Źródła podane w treści projektu
