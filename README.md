<p align="center">
  <a href="#english"><strong>English</strong></a> |
  <a href="#russian"><strong>Русский</strong></a>
</p>

<a id="english"></a>

# brainflayer-CUDA

Author: Mikhail Khoroshavin aka `XopMC`

`brainflayer-CUDA` is a CUDA tool for researching and recovering your own cryptocurrency wallets from brainwallet candidates and base private keys.

The main mode is brainwallet mode. You do not need to pass a mode flag for it. The extra `-priv` flag switches the program to base private-key mode.

This release is intentionally focused: brainwallet candidates, base private keys, sequential ranges, file and standard-input processing, GPU mask generation, Bloom/XOR filters, direct hash targets, multi-GPU execution, and practical output saving.

## Important Notice

This project is provided only for security research, testing, and recovering wallets that you own or are explicitly authorized to recover.

You are fully responsible for how you use this software. The author is not responsible for any loss, damage, legal claim, or misuse caused by running this program.

## Screenshots

### Help

![Help screenshot](docs/media/help.png)

### GPU start and target checks

![GPU target screenshot](docs/media/gpu-filter.png)

### Speed and statistics

![Speed screenshot](docs/media/statistics.png)

### Test hit

![Found test screenshot](docs/media/found.png)

## Main Features

- CUDA implementation for high-speed brainwallet and private-key checks.
- Windows and Linux builds.
- Visual Studio project and Linux Makefile included.
- Multi-GPU support through `-device 0`, `-device 0,1,3`, or `-device 0-3`.
- Brainwallet hashing: `-sha256`, `-sha3`, `-keccak`, `-blake2b`, `-raw`.
- Iteration lists for brainwallet hashing: `-iter 1,4,6-10`.
- File input, directory input, standard input, hexadecimal input with `-hex`.
- Sequential ranges with `-start`, `-end`, `-step`, `-back`, `-both`, and `-n`.
- GPU mask brute force for brainwallet candidates.
- Bloom filters, XOR filters, CPU-side Bloom/XOR post-checks, and direct hash target checks.
- `-save` writes found hashes as formatted cryptocurrency addresses.
- `-silent` hides found lines in the console but still keeps file saving working.

## Build

### Windows

Install Visual Studio and CUDA Toolkit 12.8.

```powershell
msbuild Brainflayer-CUDA.sln /p:Configuration=Release /p:Platform=x64
```

The Windows binary is created at:

```text
x64\Release\Brainflayer-CUDA.exe
```

### Linux

Install CUDA Toolkit 12.8, GCC, G++, and Make.

```bash
make CUDA_PATH=/usr/local/cuda-12.8
```

The Linux binary is created at:

```text
bin/Brainflayer-CUDA
```

The project is configured for these CUDA architectures:

```text
sm_61, sm_75, sm_86, sm_89, sm_120
```

## Basic Usage

Brainwallet mode is the default mode:

```powershell
Brainflayer-CUDA.exe -i brain.txt -c cus -bf targets.blf -save -o result.txt
```

Private-key mode:

```powershell
Brainflayer-CUDA.exe -priv -i keys.txt -hex -c c -bf targets.blf
```

Sequential private-key range:

```powershell
Brainflayer-CUDA.exe -priv -start 1 -end ffffff -step 1 -c c -hash HASH
```

Brainwallet sequential range:

```powershell
Brainflayer-CUDA.exe -start 1 -end ffff -sha256 -iter 1,2,4 -c c -bf targets.blf
```

Multi-GPU run:

```powershell
Brainflayer-CUDA.exe -i brain.txt -device 0,1,3 -c cus -bf targets.blf -save
```

GPU mask brute force:

```powershell
Brainflayer-CUDA.exe -mask pass?d?d?d -sha256 -c c -bf targets.blf
```

Custom mask charset:

```powershell
Brainflayer-CUDA.exe -cs1 abcDEF123 -mask key?1?1?1?1 -c u -hash HASH
```

## Input Sources

If no input source is selected, the program reads from standard input.

```powershell
type brain.txt | Brainflayer-CUDA.exe -c c -bf targets.blf
```

Supported sources:

```text
-i FILE          read candidates from a file
-f DIR           read candidates from files in a directory
-all             with -f, include all files instead of only text files
-delete          delete processed input files
-hex             decode every input line from hexadecimal before processing
-start/-end      generate a sequential range
-random          generate random candidates
-mask            generate brainwallet candidates on the GPU from a mask
-mask-file       read masks from a file
```

## Brainwallet Mode

Brainwallet mode is active by default.

Hashing flags:

```text
-sha256          SHA-256 brain hash, default
-sha3            SHA3-256 brain hash
-keccak          Keccak-256 brain hash
-blake2b         BLAKE2b-256 brain hash
-raw             use input bytes as a 32-byte scalar
-iter LIST       iterations list, for example 1,4,6-10
```

Only one hash flag can be active at a time. For `-raw`, the effective iteration must be `1`.

## Private-Key Mode

Use `-priv` to switch from brainwallet mode to base private-key mode.

Private keys can be read from standard input, files, directories, hexadecimal input, sequential ranges, or random generation.

```powershell
Brainflayer-CUDA.exe -priv -hex -i keys.txt -c c -bf targets.blf
```

Private-key sequential values are 256-bit values. Short values are left-padded with zero.

## Sequential Mode

Sequential mode is available in both default brainwallet mode and `-priv` mode.

```text
-start VALUE     start point
-end VALUE       end point
-step VALUE      step, default 1
-back            scan backwards
-both            scan both directions around start, requires -n
-random          random branch inside range, use with -n
-n N             candidate limit
```

For `-priv`, values are 64 hexadecimal characters internally.

For default brainwallet mode, values are 512 hexadecimal characters internally.

## Mask Brute Mode

Mask mode creates brainwallet candidates directly on the GPU. This avoids a slow CPU-to-GPU candidate pipeline for generated masks.

Built-in mask tokens:

```text
?l               lowercase letters
?u               uppercase letters
?d               digits
?h               lowercase hexadecimal
?H               uppercase hexadecimal
?s               space and ASCII symbols
?a               printable ASCII
??               literal question mark
?1 ?2 ?3 ?4      custom charsets from -cs1, -cs2, -cs3, -cs4
```

Examples:

```powershell
Brainflayer-CUDA.exe -mask pass?d?d?d -sha256 -c c -bf targets.blf
Brainflayer-CUDA.exe -mask admin?l?l?d -iter 1,2,4 -c cus -save
Brainflayer-CUDA.exe -mask-file masks.txt -n 1000000 -c c -bf targets.blf
```

## Filters and Targets

GPU filters:

```text
-bf PATH         Bloom filter
-xc PATH         compressed XOR filter
-xu PATH         uncompressed XOR filter
-xuc PATH        ultra-compressed XOR filter
-xh PATH         HC XOR filter
```

CPU post-check filters:

```text
-xx PATH         CPU XOR uncompressed verification
-xb PATH         CPU Bloom verification
```

Direct target:

```text
-hash HEX        direct hash target
-target HEX      alias for -hash
```

## Target Types

`-c` selects which target families are calculated and checked. Several letters can be used together.

```text
c - BTC compressed hash160
u - BTC uncompressed hash160
s - BTC SegWit hash160
r - BTC Taproot hash
e - Ethereum address
x - secp256k1 public-key X coordinate
t - TON popular variants
T - TON all variants
S - Solana
d - DOT, ed25519 and sr25519
f - Filecoin
i - IOTA, ed25519 and secp256k1
A - Aptos
U - SUI
X - XRP
I - ICP
Z - XTZ
```

Default target set:

```text
cus
```

## Saving Results

Without `-save`, the program writes matched hash payloads.

With `-save`, it writes formatted cryptocurrency addresses for the selected target type.

```powershell
Brainflayer-CUDA.exe -i brain.txt -c cus -bf targets.blf -save -o found.txt
```

`-silent` hides found lines in the console, but does not disable saving:

```powershell
Brainflayer-CUDA.exe -i brain.txt -c c -bf targets.blf -save -silent -o found.txt
```

## Donation

If this project helped you, you can support development:

```text
ETH:    0xDE85c1Ef7874A1D94578f11332e8fa9A6a0eE853
BTC:    bc1q063pks7ex93eka56zyumvutdt6zs9dj959pe9p
LTC:    ltc1qysumht4lxafwvmcu4ruxzuztc2xmj8tz986fmm
TRX:    TTZ3oL16BVNzU46MSJvaoKYAhvtwdTUcnz
TON:    UQC7eqLN_NlVz82YzsjzAo4iOzKjH3t095-CMtqTJ5aoqo0l
DOT:    1jen89F5v6TbdQsRaKxsCqhNp9qAdeHeZyEUWjgrM8mW6hs
DASH:   Xms41jaD967XMf2FAfEwGUxYKKhYQuok9T
SOLANA: BvDQDEgq3kbNT7VQFQRQPjc4Ta5k7d5s7GdcgoKnq3KG
```

---

<a id="russian"></a>

# brainflayer-CUDA

Автор: Михаил Хорошавин, также известен как `XopMC`

`brainflayer-CUDA` - это программа на CUDA для исследования и восстановления собственных криптовалютных кошельков по кандидатам brainwallet и по обычным закрытым ключам.

Основной режим - brainwallet. Для него не нужен отдельный флаг. Флаг `-priv` переключает программу в режим обычных закрытых ключей.

В этом выпуске оставлено только то, что нужно для практической работы: кандидаты brainwallet, обычные закрытые ключи, последовательные диапазоны, файлы и поток ввода, перебор по маске на видеокарте, Bloom/XOR фильтры, прямые цели по хешу, несколько видеокарт и нормальное сохранение результатов.

## Важное предупреждение

Проект предназначен только для исследования безопасности, проверки и восстановления кошельков, которые принадлежат вам или на восстановление которых у вас есть явное разрешение.

Вы полностью отвечаете за то, как используете эту программу. Автор не несет ответственности за потери, ущерб, претензии, нарушение закона или любое другое последствие использования программы.

## Скриншоты

### Справка

![Скриншот справки](docs/media/help.png)

### Запуск видеокарты и проверка цели

![Скриншот запуска](docs/media/gpu-filter.png)

### Скорость и статистика

![Скриншот статистики](docs/media/statistics.png)

### Тестовое совпадение

![Скриншот найденного результата](docs/media/found.png)

## Возможности

- Быстрая проверка brainwallet и закрытых ключей на CUDA.
- Сборка под Windows и Linux.
- В комплекте проект Visual Studio и Makefile для Linux.
- Работа на одной или нескольких видеокартах через `-device 0`, `-device 0,1,3` или `-device 0-3`.
- Хеширование brainwallet: `-sha256`, `-sha3`, `-keccak`, `-blake2b`, `-raw`.
- Списки повторов хеширования: `-iter 1,4,6-10`.
- Ввод из файла, папки, стандартного потока, а также шестнадцатеричный ввод через `-hex`.
- Последовательные диапазоны через `-start`, `-end`, `-step`, `-back`, `-both` и `-n`.
- Перебор brainwallet по маске прямо на видеокарте.
- Bloom фильтры, XOR фильтры, дополнительная проверка на процессоре через Bloom/XOR и прямое сравнение с хешем.
- `-save` сохраняет найденные хеши в виде адресов нужных валют.
- `-silent` скрывает найденные строки в консоли, но не мешает записи в файл.

## Сборка

### Windows

Нужно установить Visual Studio и CUDA Toolkit 12.8.

```powershell
msbuild Brainflayer-CUDA.sln /p:Configuration=Release /p:Platform=x64
```

Готовый файл будет здесь:

```text
x64\Release\Brainflayer-CUDA.exe
```

### Linux

Нужно установить CUDA Toolkit 12.8, GCC, G++ и Make.

```bash
make CUDA_PATH=/usr/local/cuda-12.8
```

Готовый файл будет здесь:

```text
bin/Brainflayer-CUDA
```

Проект собирается под такие архитектуры CUDA:

```text
sm_61, sm_75, sm_86, sm_89, sm_120
```

## Быстрый запуск

Режим brainwallet включен по умолчанию:

```powershell
Brainflayer-CUDA.exe -i brain.txt -c cus -bf targets.blf -save -o result.txt
```

Режим закрытых ключей:

```powershell
Brainflayer-CUDA.exe -priv -i keys.txt -hex -c c -bf targets.blf
```

Последовательный диапазон закрытых ключей:

```powershell
Brainflayer-CUDA.exe -priv -start 1 -end ffffff -step 1 -c c -hash HASH
```

Последовательный диапазон brainwallet:

```powershell
Brainflayer-CUDA.exe -start 1 -end ffff -sha256 -iter 1,2,4 -c c -bf targets.blf
```

Несколько видеокарт:

```powershell
Brainflayer-CUDA.exe -i brain.txt -device 0,1,3 -c cus -bf targets.blf -save
```

Перебор по маске на видеокарте:

```powershell
Brainflayer-CUDA.exe -mask pass?d?d?d -sha256 -c c -bf targets.blf
```

Своя таблица символов для маски:

```powershell
Brainflayer-CUDA.exe -cs1 abcDEF123 -mask key?1?1?1?1 -c u -hash HASH
```

## Источники кандидатов

Если источник не указан, программа читает строки из стандартного потока.

```powershell
type brain.txt | Brainflayer-CUDA.exe -c c -bf targets.blf
```

Поддерживаемые источники:

```text
-i FILE          читать кандидаты из файла
-f DIR           читать кандидаты из файлов в папке
-all             вместе с -f читать все файлы, а не только текстовые
-delete          удалять обработанные входные файлы
-hex             считать каждую строку шестнадцатеричными байтами
-start/-end      создать последовательный диапазон
-random          создать случайные кандидаты
-mask            создать кандидаты brainwallet по маске на видеокарте
-mask-file       читать маски из файла
```

## Режим brainwallet

Этот режим включен по умолчанию.

Флаги хеширования:

```text
-sha256          SHA-256, используется по умолчанию
-sha3            SHA3-256
-keccak          Keccak-256
-blake2b         BLAKE2b-256
-raw             использовать входные байты как 32-байтовое число
-iter LIST       список повторов, например 1,4,6-10
```

Одновременно можно выбрать только один способ хеширования. Для `-raw` допустим только один проход.

## Режим закрытых ключей

Флаг `-priv` переключает программу из режима brainwallet в режим обычных закрытых ключей.

Закрытые ключи можно читать из стандартного потока, файлов, папок, шестнадцатеричного ввода, последовательных диапазонов или случайной генерации.

```powershell
Brainflayer-CUDA.exe -priv -hex -i keys.txt -c c -bf targets.blf
```

Значения последовательного перебора для `-priv` - это 256-битные числа. Короткие значения дополняются нулями слева.

## Последовательный перебор

Последовательный перебор работает и в режиме brainwallet, и в режиме `-priv`.

```text
-start VALUE     начало диапазона
-end VALUE       конец диапазона
-step VALUE      шаг, по умолчанию 1
-back            идти назад
-both            идти в обе стороны от start, требует -n
-random          случайные значения внутри диапазона, использовать с -n
-n N             ограничение количества кандидатов
```

Для `-priv` значения внутри программы имеют длину 64 шестнадцатеричных символа.

Для режима brainwallet значения внутри программы имеют длину 512 шестнадцатеричных символов.

## Перебор по маске

Режим маски создает кандидаты brainwallet прямо на видеокарте. Это убирает медленную передачу сгенерированных строк с процессора на видеокарту.

Готовые обозначения:

```text
?l               строчные латинские буквы
?u               заглавные латинские буквы
?d               цифры
?h               шестнадцатеричные символы в нижнем регистре
?H               шестнадцатеричные символы в верхнем регистре
?s               пробел и печатные символы
?a               все печатные ASCII-символы
??               обычный знак вопроса
?1 ?2 ?3 ?4      свои наборы символов из -cs1, -cs2, -cs3, -cs4
```

Примеры:

```powershell
Brainflayer-CUDA.exe -mask pass?d?d?d -sha256 -c c -bf targets.blf
Brainflayer-CUDA.exe -mask admin?l?l?d -iter 1,2,4 -c cus -save
Brainflayer-CUDA.exe -mask-file masks.txt -n 1000000 -c c -bf targets.blf
```

## Фильтры и цели

Фильтры на видеокарте:

```text
-bf PATH         Bloom фильтр
-xc PATH         XOR фильтр для compressed
-xu PATH         XOR фильтр для uncompressed
-xuc PATH        ultra-compressed XOR фильтр
-xh PATH         HC XOR фильтр
```

Дополнительная проверка на процессоре:

```text
-xx PATH         проверка XOR uncompressed на процессоре
-xb PATH         проверка Bloom на процессоре
```

Прямая цель:

```text
-hash HEX        прямое сравнение с хешем
-target HEX      то же самое, что -hash
```

## Типы целей

Флаг `-c` выбирает, какие семейства целей считать и проверять. Можно указывать несколько букв вместе.

```text
c - BTC compressed hash160
u - BTC uncompressed hash160
s - BTC SegWit hash160
r - BTC Taproot hash
e - Ethereum address
x - secp256k1 public-key X coordinate
t - TON popular variants
T - TON all variants
S - Solana
d - DOT, ed25519 and sr25519
f - Filecoin
i - IOTA, ed25519 and secp256k1
A - Aptos
U - SUI
X - XRP
I - ICP
Z - XTZ
```

По умолчанию используется:

```text
cus
```

## Сохранение результатов

Без `-save` программа выводит найденные хеши.

С `-save` программа выводит и сохраняет адреса валют для выбранного типа цели.

```powershell
Brainflayer-CUDA.exe -i brain.txt -c cus -bf targets.blf -save -o found.txt
```

`-silent` скрывает найденные строки в консоли, но не отключает сохранение:

```powershell
Brainflayer-CUDA.exe -i brain.txt -c c -bf targets.blf -save -silent -o found.txt
```

## Донат

Если проект оказался полезен, можно поддержать разработку:

```text
ETH:    0xDE85c1Ef7874A1D94578f11332e8fa9A6a0eE853
BTC:    bc1q063pks7ex93eka56zyumvutdt6zs9dj959pe9p
LTC:    ltc1qysumht4lxafwvmcu4ruxzuztc2xmj8tz986fmm
TRX:    TTZ3oL16BVNzU46MSJvaoKYAhvtwdTUcnz
TON:    UQC7eqLN_NlVz82YzsjzAo4iOzKjH3t095-CMtqTJ5aoqo0l
DOT:    1jen89F5v6TbdQsRaKxsCqhNp9qAdeHeZyEUWjgrM8mW6hs
DASH:   Xms41jaD967XMf2FAfEwGUxYKKhYQuok9T
SOLANA: BvDQDEgq3kbNT7VQFQRQPjc4Ta5k7d5s7GdcgoKnq3KG
```
