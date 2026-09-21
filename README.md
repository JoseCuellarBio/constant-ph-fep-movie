# Constant-pH Monte Carlo sobre una trayectoria molecular

Este directorio contiene un flujo de trabajo para muestrear estados de
protonacion sobre los frames de una trayectoria DCD. En cada intento se elige
un residuo ionizable, se propone alternar su carga y se acepta o rechaza el
cambio mediante un criterio de Metropolis. Los estados aceptados se conservan
entre intentos y entre frames. A partir de esos estados tambien puede calcularse
una energia Debye--Huckel por frame.

El codigo **no modifica las coordenadas ni escribe una nueva trayectoria**: la
"perturbacion" es un cambio del estado de carga asociado a cada residuo. Los
resultados se guardan como tablas CSV que pueden usarse en un analisis posterior
o como entrada para otro flujo de FEP.

## Contenido

| Archivo | Funcion |
|---|---|
| `Montecarlo_2.py` | Modelo de la proteina, seleccion de residuos, terminos energeticos y aceptacion de Metropolis. |
| `periodic_neighborhood.py` | Vecindades con imagen minima para aplicar la celda periodica del DCD al Monte Carlo. |
| `montecarlo_pH_attempt_md_dcd.py` | Flujo directo: muestrea protonaciones y calcula la energia final en una sola ejecucion. |
| `montecarlo_questions_md_dcd.py` | Etapa 1 del flujo separado: solo muestrea y guarda los intentos. |
| `debye_huckel_from_mc_csv.py` | Etapa 2 del flujo separado: reconstruye los estados visitados y calcula energias. |
| `montecarlo_pH_out.csv` | Ejemplo de estados finales y energia por frame. |
| `montecarlo_pH_attempts.csv` | Ejemplo del historial de intentos Monte Carlo. |

## Requisitos

- Python 3.10 o posterior (se usan anotaciones de tipo con `|`).
- NumPy.
- MDTraj.
- Un PDB que defina la topologia.
- Un DCD con la misma cantidad y el mismo orden de atomos que el PDB.

Instalacion minima en un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy mdtraj
```

Los numeros de residuo (`resSeq`) deben ser unicos en toda la topologia,
incluidas las distintas cadenas. El programa se detiene si encuentra numeros
repetidos. MDTraj entrega las coordenadas en nm; todas las distancias internas
del flujo DCD se interpretan en esa unidad.

## Uso rapido: muestreo y energia en una sola etapa

Ejecutar desde este directorio:

```bash
python montecarlo_pH_attempt_md_dcd.py \
  --pdb estructura.pdb \
  --dcd trayectoria.dcd \
  --ph 7.0 \
  --attempts-per-frame 100 \
  --seed 123 \
  --output estados_finales.csv \
  --attempts-output intentos.csv
```

El script procesa el DCD por bloques para no cargar toda la trayectoria en
memoria. Sus opciones son:

| Opcion | Predeterminado | Descripcion |
|---|---:|---|
| `--pdb` | `16_allatoms.pdb` | Topologia PDB. |
| `--dcd` | `16_allatoms_wrapped.dcd` | Trayectoria DCD. |
| `--ph` | `7.0` | pH usado en el termino quimico. |
| `--attempts-per-frame` | `10` | Intentos secuenciales por frame. |
| `--seed` | sin semilla | Semilla de Python y NumPy para reproducibilidad. |
| `--chunk` | `100` | Numero de frames leidos por bloque. |
| `--pbc-mode` | `auto` | Seleccion de la fuente de la caja periodica. |
| `--box-lengths LX LY LZ` | ninguna | Caja ortorrombica indicada por el usuario, en nm. |
| `--bounds-padding` | `0.1` | Margen por lado para la caja estimada desde coordenadas, en nm. |
| `--output` | `montecarlo_pH_out.csv` | Estado final y energia por frame. |
| `--attempts-output` | `montecarlo_pH_attempts.csv` | Historial completo de propuestas. |

Los nombres de entrada predeterminados no estan incluidos actualmente en este
directorio; por eso, en general deben indicarse `--pdb` y `--dcd`.

## Seleccion de condiciones periodicas

`--pbc-mode` permite decidir como se obtienen los vectores de celda. La misma
seleccion se aplica a los intentos Monte Carlo y al calculo Debye--Huckel.

| Modo | Comportamiento |
|---|---|
| `auto` | Primero usa la caja guardada en el DCD; si no existe, usa `--box-lengths`; como ultimo recurso estima una caja con los limites de las coordenadas. |
| `trajectory` | Exige una caja valida en el DCD y falla si no esta disponible. |
| `box` | Exige `--box-lengths LX LY LZ` y construye una caja ortorrombica fija. |
| `bounds` | Calcula por frame `L = max(xyz) - min(xyz) + 2*padding`. |
| `none` | Desactiva las condiciones periodicas y usa distancias euclideas directas. |

Ejemplos:

```bash
# Usar exclusivamente la celda del DCD
python montecarlo_pH_attempt_md_dcd.py --pdb estructura.pdb \
  --dcd trayectoria.dcd --pbc-mode trajectory

# Caja ortorrombica fija de 10 x 10 x 12 nm
python montecarlo_pH_attempt_md_dcd.py --pdb estructura.pdb \
  --dcd trayectoria.dcd --pbc-mode box --box-lengths 10 10 12

# Inferir una caja por frame y agregar 0.2 nm a cada lado
python montecarlo_pH_attempt_md_dcd.py --pdb estructura.pdb \
  --dcd trayectoria.dcd --pbc-mode bounds --bounds-padding 0.2

# Ignorar por completo las condiciones periodicas
python montecarlo_pH_attempt_md_dcd.py --pdb estructura.pdb \
  --dcd trayectoria.dcd --pbc-mode none
```

El modo `bounds` es una aproximacion: mide la extension ocupada por los atomos,
no necesariamente la caja fisica original. Es mas razonable con solvente
explicito que llena la celda y puede producir resultados artificiales si el DCD
solo contiene la proteina. El margen evita que los atomos extremos queden
identificados exactamente como copias periodicas.

## Flujo recomendado para separar muestreo y evaluacion energetica

Separar las etapas permite calcular las energias despues del muestreo y analizar
tanto el estado final como el promedio de los estados visitados.

### 1. Generar estados de protonacion

```bash
python montecarlo_questions_md_dcd.py \
  --pdb estructura.pdb \
  --dcd trayectoria.dcd \
  --ph 7.0 \
  --attempts-per-frame 100 \
  --seed 123 \
  --output estados_finales.csv \
  --attempts-output intentos.csv
```

Este programa exige de forma explicita las cuatro rutas de entrada/salida. Las
opciones `--no-polar` y `--no-electrostatic` permiten desactivar, de manera
independiente, esos terminos del criterio de aceptacion:

```bash
python montecarlo_questions_md_dcd.py \
  --pdb estructura.pdb --dcd trayectoria.dcd \
  --output estados_finales.csv --attempts-output intentos.csv \
  --no-electrostatic
```

### 2. Calcular las energias de los estados visitados

```bash
python debye_huckel_from_mc_csv.py \
  --pdb estructura.pdb \
  --dcd trayectoria.dcd \
  --charges-csv estados_finales.csv \
  --attempts-csv intentos.csv \
  --output energias.csv
```

La segunda etapa valida que el DCD y ambos CSV contengan los mismos frames, que
los intentos esten ordenados y que el estado reconstruido coincida con el estado
final guardado. Deben pasarse las mismas opciones `--pbc-mode`, `--box-lengths`
y `--bounds-padding` usadas durante el muestreo para evaluar exactamente la
misma geometria periodica.

## Modelo de protonacion

Los residuos titulables y sus cargas permitidas son:

| Tipo | Residuos | Estado cargado | Estado neutro |
|---|---|---:|---:|
| Acido | ASP, GLU, CYS, TYR, CTR | -1 | 0 |
| Basico | ARG, HIS, LYS, NTR | +1 | 0 |

Al iniciarse, todos esos residuos se colocan en su estado cargado. Cada intento
elige uno al azar y propone alternar entre el estado cargado y el neutro. Los
valores de pKa usados son: ASP 4.0, GLU 4.5, HIS 6.4, CYS 8.3, TYR 11.0, LYS
10.6, ARG 12.0, NTR 7.5 y CTR 3.5.

El cambio energetico de una propuesta es

```text
Delta E = Delta E_pH + Delta E_electrostatica + Delta E_polar
```

con

```text
Delta E_pH = Delta q (pH - pKa) kB T ln(10)
```

donde `T = 300 K` y `kB = 0.001987 kcal mol^-1 K^-1`. La contribucion
electrostatica es una interaccion apantallada con longitud de apantallamiento de
1 nm; la contribucion polar depende de la cantidad y el tipo de vecinos. Una
propuesta con `Delta E < 0` se acepta siempre; de otro modo se acepta con
probabilidad `exp(-Delta E / kB T)`.

Para representar geometricamente cada residuo se usa `CB`; si no existe, `CA`,
y si tampoco existe, `O`. Cuando el DCD contiene vectores de celda validos, las
distancias de vecindad se calculan con la convencion de imagen minima. La misma
transformacion a coordenadas fraccionarias se usa en el criterio Monte Carlo y
en la energia Debye--Huckel. Si el DCD no contiene celda, ambos calculos usan
distancias directas.

## Energia Debye--Huckel informada

La energia posterior al muestreo es distinta de `Delta E`: es una magnitud de
salida calculada para el estado completo,

```text
E_DH = 0.5 sum(i<j) [q_i q_j / r_ij] exp(-r_ij / 1 nm)
```

Solo se incluyen pares cargados con `0 < r_ij < 3 nm`. Si el DCD contiene
vectores de celda validos, en este calculo si se aplica la convencion de imagen
minima; sin celda se usan distancias directas. El resultado se etiqueta en
`kcal/mol`, siguiendo el prefactor fijo implementado en los scripts.

## Formatos de salida

### Estado final por frame

El flujo directo produce:

```text
frame,attempted_residue,accepted,debye_huckel_energy_kcal_mol,debye_huckel_visited_mean_kcal_mol,visited_states,ASP1,GLU3,...
```

En el flujo directo, `debye_huckel_energy_kcal_mol` es la energia del estado
final y `debye_huckel_visited_mean_kcal_mol` es el promedio de las energias
posteriores a cada intento del frame. Los intentos rechazados tambien se
contabilizan y repiten la energia del estado anterior. `visited_states` coincide
con `--attempts-per-frame`; el estado inicial previo al primer intento no se
agrega como una observacion independiente.

El flujo separado omite las columnas de energia en su primera etapa. En ambos
flujos, `attempted_residue` y `accepted` describen **solo el ultimo intento del
frame**. Las columnas `ASP1`, `GLU3`, etc. contienen el estado final de todos los
residuos titulables despues de completar los intentos de ese frame.

### Historial de intentos

```text
frame,attempt,residue,old_charge,proposed_charge,accepted,resulting_charge
0,1,ASP41,-1.0,0.0,0,-1.0
```

- `attempt` comienza en 1 dentro de cada frame.
- `accepted` vale 1 cuando se acepta la propuesta y 0 cuando se rechaza.
- En un rechazo, `resulting_charge` debe coincidir con `old_charge`.

### Energia del flujo separado

```text
frame,debye_huckel_final_kcal_mol,debye_huckel_visited_mean_kcal_mol,visited_states
```

- `debye_huckel_final_kcal_mol`: energia del estado al terminar el frame.
- `debye_huckel_visited_mean_kcal_mol`: promedio de la energia despues de cada
  intento, incluidos los intentos rechazados (que repiten el estado anterior).
- `visited_states`: numero de intentos contabilizados en el promedio.

## Reproducibilidad y consideraciones

- Use siempre `--seed` para repetir exactamente la secuencia Monte Carlo con la
  misma version del codigo y los mismos archivos.
- Aumentar `--attempts-per-frame` mejora el muestreo dentro de cada geometria,
  pero tambien permite que el estado evolucione mas antes del siguiente frame.
- Los estados no se reinician al cambiar de frame: el primer intento de un frame
  parte del estado final del frame anterior.
- `NTR` y `CTR` solo se reconocen si aparecen como nombres de residuo en la
  topologia; el codigo no crea terminales titulables automaticamente.
- Los CSV de ejemplo pueden ser grandes porque contienen una columna por cada
  residuo titulable.

Para consultar la interfaz exacta de un programa:

```bash
python montecarlo_pH_attempt_md_dcd.py --help
python montecarlo_questions_md_dcd.py --help
python debye_huckel_from_mc_csv.py --help
```
