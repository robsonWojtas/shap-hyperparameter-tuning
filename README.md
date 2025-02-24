# shap-hyperparameter-tuning

# Projekt Magisterski

**Autor:** Robert Wojtaś  
**Temat pracy:** Badanie przydatności metody SHAP w procesie strojenia hiperparametrów metod uczenia maszynowego


## Streszczenie

Niniejsza praca magisterska skupia się na analizie i ocenie potencjału metody
SHAP (ang. SHapley Additive exPlanations) w kontekście strojenia hiperparametrów metod
uczenia maszynowego. Metoda SHAP wywodząca się z obszaru XAI (ang. Explainable
Artificial Intelligence), pozwala na lepszą interpretację wyników generowanych przez
modele uczenia maszynowego, uwydatniając wkład poszczególnych cech w proces decyzyjny modelu. W pracy przedstawiono nowatorskie podejście do wykorzystania metody
SHAP w procesie optymalizacji hiperparametrów. Zamiast tradycyjnych metod, takich jak
hipersiatka czy przeszukiwanie losowe, zaproponowano sposób wykorzystania metody
SHAP, w którym proces strojenia hiperparametrów opiera się na informacjach uzyskanych
z SHAP, pomagających identyfikować, które hiperparametry mają największy wpływ na
jakość modelu w kontekście konkretnego problemu.


## Struktura projektu

## Wymagania techniczne

- **Python:** Wersja 3.9
- **Biblioteki:**
    - scikit-learn
    - matplotlib
    - numpy
    - pandas
    - SHAP
    - (inne zależności wymienione w `requirements.txt`)
- **Środowisko:**
    - Wirtualne środowisko (`venv`)
    - Narzędzia do zarządzania wersjami (np. Git)

## Praca z projektem

1. **Utworzenie i aktywacja wirtualnego środowiska:**
- W systemie Linux:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
- W systemie Windows:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
2. **Instalacja zależności:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Uruchomienie poszczególnych skryptów:**
    ```bash
    python <nazwa_skryptu>.py
    ```
   
## Zbiory danych

- **Iris:** Zbiór danych zawierający 150 próbek trzech gatunków Irysów (Iris setosa, Iris virginica, Iris versicolor). Każda próbka zawiera 4 cechy: długość i szerokość płatków oraz długość i szerokość działki kielicha. Pobierany z biblioteki scikit-learn.
- **Wine:** Zbiór danych zawierający 178 próbek trzech gatunków win (klasy 0, 1, 2). Każda próbka zawiera 13 cech chemicznych wina. Pobierany z biblioteki scikit-learn.
- **Breast Cancer:** Zbiór danych zawierający 569 próbek nowotworów piersi. Każda próbka zawiera 30 cech opisujących charakterystykę komórek nowotworowych. Pobierany z biblioteki scikit-learn.
- **Diabetes:** Zbiór danych zawierający 768 próbki pacjentów z cukrzycą. Każda próbka zawiera 10 cech pacjentów. Ładowany z pliku CSV. Źródło: https://www.kaggle.com/datasets/mathchi/diabetes-data-set
- **California Housing:** Zbiór danych zawierający 20640 próbek cen nieruchomości w Kalifornii. Każda próbka zawiera 8 cech opisujących nieruchomość. Ładowany z pliku CSV. Źródło: https://www.kaggle.com/datasets/camnugent/california-housing-prices
- **Adult Census:** Zbiór danych zawierający 48842 próbki danych demograficznych dorosłych. Każda próbka zawiera 14 cech opisujących daną osobę. Ładowany z pliku CSV. Źródło: https://www.kaggle.com/datasets/wenruliu/adult-income-dataset