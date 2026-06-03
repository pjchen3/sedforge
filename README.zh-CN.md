# sedforge

<p align="right">
  <a href="README.md"><img src="https://img.shields.io/badge/English-README-2563eb" alt="English README"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-README-c2410c" alt="简体中文 README"></a>
</p>

sedforge 是一个用于恒星光度 SED（spectral energy distribution）拟合的Python 包。它以绝对通量为拟合对象，使用 MCMC 采样，支持单星、未分辨双星和未分辨三组分系统。

本项目是 [Speedyfit](https://github.com/vosjo/speedyfit) 的科研 fork。在原始 Speedyfit 的基础上，sedforge 对输入文件、模型网格、消光轴、绘图、catalog photometry helper 等做了较多调整，以适配下面描述的“星等输入优先”的工作流。

主要特性：

- 从简单的 `photband mag mag_err system` 光度表读取观测数据，并在内部转换为与模型网格一致的 band-averaged `Flambda`；
- `distance` 是以 parsec 为单位的物理拟合参数或固定参数；
- 当积分模型网格提供 `[Fe/H]` 轴时，可以把金属丰度作为真实网格轴拟合；
- 任意模型参数都可以通过 YAML 中的 `fixed:` 部分固定；
- 构建积分网格时，先在每个波长点应用消光，再通过滤光片响应曲线积分；
- 内置滤光片响应曲线来自 SVO Filter Profile Service，并在
  `filter_info.dat` 中记录 photon/energy response convention；
- 默认消光律为 `WC2019`，默认 `case1=1`。

本 fork 继承原 Speedyfit 的 GPLv3 license。公开发布或再分发时，请保留
GPLv3 license 和原始 attribution。

## 安装

sedforge 需要 Python 3.9 或更新版本。在仓库根目录下安装：

```bash
cd sedforge
python -m pip install .
```

开发模式安装，包括测试和构建工具：

```bash
python -m pip install -e ".[dev]"
```

可选依赖被放在 extras 中，避免把非核心依赖装进默认环境：

```bash
python -m pip install ".[photometry]"  # 使用 astroquery 下载 VizieR photometry
python -m pip install ".[svo]"         # 更新 SVO 滤光片曲线的辅助脚本
python -m pip install ".[hdf5]"        # HDF5 模型网格支持
```

模型网格目录通过环境变量 `SEDFORGE_MODELS` 指定：

```bash
export SEDFORGE_MODELS=/path/to/sed_models
```

这个目录应包含 `grid_description.yaml` 以及其中引用的模型文件。推荐的本地布局：

```text
sed_models/
  grid_description.yaml
  raw/              # 原始模型光谱
  integrated/       # 通过滤光片积分后的拟合网格
  spectral_cache/   # 只用于绘图的连续光谱缓存
```

大型模型网格不应提交到 GitHub。建议把它们保存在本地、服务器、GitHub
Releases、Zenodo 或 Figshare，并在论文或 README 中说明生成方式和下载位置。

## 快速开始

先创建一个星等光度文件，必须包含以下列：

```text
photband  mag    mag_err  system
GAIA3E_G  12.30  0.01     vega
PS1_g     18.42  0.02     ab
2MASS_Ks  10.10  0.02     vega
```

`photband` 需要匹配包内的滤光片响应曲线名称，例如 `GAIA3E_G`、
`2MASS_Ks`、`WISE_RSR_W1`、`HST_WFC3_F814W`，或
`sedforge/transmission_curves` 中的其他波段。

生成一个起始 setup 文件并运行拟合：

```bash
sedforge setup my_target -grid ck_all
sedforge fit my_target_setup_ck_all.yaml --noplot
```

默认情况下，拟合会在 setup 文件旁边寻找 `my_target.phot`。输出通常包括：

- 一行 CSV 拟合结果摘要；
- FITS 格式的 accepted MCMC samples；
- SED 拟合图；
- posterior corner plot。

## 光度输入格式

推荐输入格式为星等表：

```text
photband  mag    mag_err  system
GAIA3E_G  12.30  0.01     vega
PS1_g     18.42  0.02     ab
2MASS_Ks  10.10  0.02     vega
```

sedforge 会使用与模型网格相同的 SVO 响应曲线，把星等转换为
band-averaged `Flambda`。`sedforge photometry` 生成的文件会保留内部转换后的
flux 列，便于检查：

```text
photband  mag    mag_err  system  mag_type  mag_zp_offset  flux       flux_err
GAIA3E_G  12.30  0.01     vega    pogson    0.00           1.23e-13  1.13e-15
```

如果同一个表里同时有 magnitude 和 flux 列，fitter 会优先使用 magnitude
列并重新计算 flux。高级输入仍可使用 `photband flux flux_err`，但这些 flux
必须已经是 band-averaged `erg/s/cm2/Angstrom`。

常见内置滤光片的默认星等系统：

- Vega: `GAIA3E`, `2MASS`, `WISE_RSR`, `SPITZER_IRAC`, `WFCAM`
- AB: `GALEX`, `PS1`, `SDSS`, `SkyMapper`, `ZTF`

HST 滤光片没有默认系统，因为同一 passband 可能用 VegaMag、ABMag 或 STMag
报告。HST photometry 应显式提供 `system: vega` 或 `system: ab`。当前不支持
STMag 输入。

SDSS catalog magnitudes 是 luptitudes/asinh magnitudes，因此 sedforge 不用
高信噪比 Pogson 近似直接转换。对于 `SDSS_u/g/r/i/z`，代码使用 SDSS softening
parameters，并反解 asinh magnitude。若你的 SDSS 数据已经转成普通 AB/Pogson
星等，请在表中显式设置 `mag_type` 为 `pogson`，并设置 `mag_zp_offset` 为
`0.0`。

可以在 setup 文件中使用 `photband_include` 或 `photband_exclude` 选择波段。
选择器支持 family prefix，例如 `GAIA3E` 会匹配 `GAIA3E_G`、`GAIA3E_BP` 和
`GAIA3E_RP`。

## 从 VizieR 下载星等

安装 photometry extra 后，可以用坐标或 Gaia DR3 source id 从配置好的
VizieR catalogs 生成光度表。默认配置包括：

- Gaia DR3: `I/355/gaiadr3`
- 2MASS: `II/246/out`
- AllWISE: `II/328/allwise`
- Pan-STARRS1: `II/349/ps1`
- SDSS DR12: `V/147/sdss12`
- GLIMPSE: `II/293/glimpse`
- SkyMapper DR2: `II/379`
- GALEX AIS: `II/312/ais`

使用 Gaia DR3 source id：

```bash
sedforge photometry \
  --gaia-id 1234567890123456789 \
  --output my_target.phot \
  --metadata-output my_target_catalogs.dat
```

使用坐标：

```bash
sedforge photometry \
  --ra 10.6847083 --dec 41.26875 \
  --output my_target.phot
```

可通过 `--catalog-config my_catalogs.yaml` 覆盖内置 catalog 配置。

## 模型网格

积分模型网格是 FITS table。必要列包括模型轴，例如 `teff`、`logg`、`av`，
可选 `feh`，以及每个滤光片的一列 flux。`Labs` 列存储拟合输出中使用的
bolometric luminosity 信息。

模型目录由 `grid_description.yaml` 描述。一个固定金属丰度网格示例：

```yaml
ckp00:
  filename: ck03_p00
  raw_filename: raw/ck/ck03_p00
  feh: 0.0
  spectral_cache: spectral_cache/ck_all_plot_spectra.fits
  info: Castelli & Kurucz 2003, [Fe/H] = 0.0
```

一个金属丰度 stack 示例：

```yaml
ck_all:
  filename: ck_all
  integrated_subdir: integrated
  spectral_cache: spectral_cache/ck_all_plot_spectra.fits
  info: Combined Castelli & Kurucz metallicity stack
  members:
    - grid: ckm05
      feh: -0.5
    - grid: ckp00
      feh: 0.0
    - grid: ckp05
      feh: 0.5
```

如果 FITS 网格本身包含真实 `[Fe/H]` 列，可以声明：

```yaml
newera_alpha0:
  filename: newera_alpha0
  integrated_subdir: integrated
  spectral_cache: spectral_cache/newera_alpha0_plot_spectra.fits
  supports_feh: true
  info: PHOENIX NewEra alpha=0 integrated grid
```

## 绘图用原始光谱缓存

拟合使用积分网格，以保持速度和文件大小。SED 图中显示的连续模型光谱应放在
单独的 spectral cache 中。该 FITS 文件包含：

- `PARAMS`: 每个模型光谱一行，列如 `teff`、`logg`、`feh`、`he_mass`；
- `WAVE`: 公共波长网格，单位 Angstrom；
- `FLUX`: 二维 `(n_spectra, n_wave)` 数组，单位 `erg/s/cm2/Angstrom`。

fitter 会用 FITS memory mapping 打开这些文件，只读取绘图需要的最近模型光谱，
避免把整个光谱库一次性载入内存。

## 消光律与滤光片积分

积分网格生成时，sedforge 会先对每个模型光谱在每个波长点应用消光律，然后再
通过滤光片响应曲线积分。这避免了把消光近似成单一有效波长修正。

默认消光律为 WC2019，即 Wang & Chen (2019),
*The Optical to Mid-infrared Extinction Law Based on the APOGEE, Gaia DR2,
Pan-STARRS1, SDSS, APASS, 2MASS, and WISE Surveys*, ApJ, 877, 116,
doi:[10.3847/1538-4357/ab1c61](https://doi.org/10.3847/1538-4357/ab1c61)。

默认设置：

```yaml
reddening_law: WC2019
reddening_Rv: 3.1
reddening_case1: 1
```

对于普通 FITS 积分网格，`reddening_Rv` 是网格选择常数，不是 MCMC 采样参数。
不要在 `ck_all`、`newera_alpha0`、`tlusty_all`、`koester2` 或 `blackbody`
这类网格的 `pnames` 中放 `rv`。

Cepheid 相关工作可以使用特殊 HDF5 网格 `ck03_cepheid_rv`，该网格有显式
`rv` 轴。此时可以把 `rv` 作为拟合参数，或在 `fixed:` 中固定。

内置滤光片曲线由 `filter_svo_map.dat` 和 SVO Filter Profile Service 生成。
`filter_info.dat` 记录 SVO id 和本地 `response_type`：photon response 在合成
photometry 中使用额外波长权重，energy response 不使用额外波长权重。SVO 的
WISE 和 Spitzer/IRAC 曲线是 energy responses。

## Setup 文件

一个 YAML setup 文件控制一次拟合。主要内容包括：

- target 和 photometry；
- 模型网格和消光律；
- 拟合参数和硬边界；
- 固定参数；
- 拟合参数的 Gaussian priors；
- MCMC sampler 设置；
- 输出文件和绘图设置。

`pnames` 和 `limits` 定义 MCMC 采样参数。`fixed` 定义不参与采样的固定参数。
所选网格需要的每个模型参数，都必须出现在 `pnames` 或 `fixed` 中。

固定 `[Fe/H]` 示例：

```yaml
fixed:
  feh: 0.0
```

不要用相同上下限来“固定”一个参数。`pnames` 里的参数必须有真实的非零拟合
范围；固定值统一放在 `fixed` 里。

`fixed` 是硬固定值，不参与采样。`priors` 是 posterior 中的 Gaussian priors，
参数仍然参与采样，因此每个 prior 都必须对应 `pnames` 中的一个名字：

```yaml
priors:
  distance: [1000.0, 50.0]
```

派生量如 `L`、`mass`、`q` 是输出或检查量，不是拟合参数，不能放进 `priors`。

setup 文件中的消光参数始终是 `av`，即以 magnitude 为单位的 `A(V)`。旧的
`ebv` / `E(B-V)` 参数会被拒绝。

## 单星示例

下面的 setup 拟合 `teff`、`logg`、`rad`、`distance` 和 `av`，同时固定
`[Fe/H]=0.0`：

```yaml
objectname: example_single
photometryfile: example_single.phot
photband_exclude: []

grids:
  - ck_all
reddening_law: WC2019
reddening_Rv: 3.1
reddening_case1: 1

pnames: [teff, logg, rad, distance, av]
limits:
  - [5000, 9000]      # teff, K
  - [3.0, 5.0]        # logg, dex
  - [0.1, 10.0]       # radius, Rsun
  - [100, 5000]       # distance, pc
  - [0.0, 3.1]        # Av, mag

fixed:
  feh: 0.0

priors: {}

nwalkers: 80
nsteps: 1000
nrelax: 300
a: 2
percentiles: [16, 50, 84]

resultfile: example_single_results.csv
datafile: example_single_samples.fits
plot1:
  type: sed_fit
  result: pc
  path: example_single_sed.png
plot2:
  type: distribution
  show_best: true
  path: example_single_corner.png
  parameters: [teff, logg, rad, distance, av]
```

运行：

```bash
sedforge fit example_single_setup.yaml --noplot
```

## 多组分拟合

双星 setup 每个组分使用一个网格。共享参数如 `distance`、`av` 和 `feh` 只需
提供一次；第二个组分的参数使用后缀 `2`。

示例：

```yaml
grids:
  - ck_all
  - ck_all

pnames: [teff, logg, rad, teff2, logg2, rad2, distance, av]
limits:
  - [8000, 16000]     # teff
  - [3.5, 5.0]        # logg
  - [0.5, 5.0]        # rad
  - [3500, 6500]      # teff2
  - [2.0, 4.5]        # logg2
  - [1.0, 20.0]       # rad2
  - [100, 5000]       # distance
  - [0.0, 1.55]       # Av

fixed:
  feh: 0.0
```

如果两个组分应使用不同金属丰度：

```yaml
fixed:
  feh: 0.0
  feh2: -0.5
```

三组分拟合同理：提供三个网格，并使用后缀 `3` 表示第三个组分的模型参数。
网格数量必须与组分数量一致。

## 常用命令

创建起始 setup：

```bash
sedforge setup my_target -grid ck_all
```

运行拟合：

```bash
sedforge fit my_target_setup_ck_all.yaml --noplot
```

检查已安装模型网格：

```bash
sedforge checkgrids
sedforge checkgrids --bands
```

从 VizieR 下载星等光度文件：

```bash
sedforge photometry --ra 10.6847083 --dec 41.26875 \
  --output my_target.phot
```

运行测试：

```bash
python -m pytest
```

构建 source distribution 和 wheel：

```bash
python -m build
```

## 输出文件

典型输出包括：

- `resultfile`: 一行 CSV，包含 median 值以及 16th/84th percentile uncertainty；
- `datafile`: FITS table 格式的 accepted MCMC samples；
- SED plot: 观测 flux 和最佳/percentile 模型 SED；
- corner plot: 采样参数的 posterior distributions。

corner plot 的常见参数标签包含物理单位，例如 `teff`、`logg`、`feh`、`rad`、
`distance` 和 `av`。

## License 与引用

sedforge 派生自 Joris Vos 的原始
[Speedyfit](https://github.com/vosjo/speedyfit) package，并保留 GPLv3 license。
完整 license 见 `LICENSE`。

如果在论文中使用 sedforge，请引用 `CITATION.cff` 中的软件条目。由于 sedforge
派生自 Speedyfit，也请引用 Speedyfit 仓库和相关论文：

- [Speedyfit](https://github.com/vosjo/speedyfit)，原始软件仓库；
- Vos et al. (2017), *The orbits of subdwarf-B + main-sequence binaries. III.
  The period-eccentricity distribution*, A&A, 605, A109,
  doi:[10.1051/0004-6361/201730958](https://doi.org/10.1051/0004-6361/201730958)；
- Vos et al. (2018), *Composite hot subdwarf binaries - I. The
  spectroscopically confirmed sdB sample*, MNRAS, 473, 693-709,
  doi:[10.1093/mnras/stx2198](https://doi.org/10.1093/mnras/stx2198)。

还应引用分析中实际使用的模型大气网格、滤光片响应曲线/catalogs 和消光律。
例如论文中应说明：

- 模型 family 和 grid release，例如 Castelli & Kurucz、PHOENIX/NewEra、
  TLUSTY、Koester、TMAP 或 blackbody grids；
- 滤光片响应曲线来源，例如 SVO Filter Profile Service；
- 查询的 photometry catalogs，例如 Gaia DR3、2MASS、AllWISE、PS1、SDSS、
  GLIMPSE、SkyMapper 或 GALEX；
- 消光律和参数，例如 `WC2019`、`Rv` 和 `case1`。对于内置 WC2019 law，请引用
  Wang & Chen (2019),
  doi:[10.3847/1538-4357/ab1c61](https://doi.org/10.3847/1538-4357/ab1c61)。

问题反馈、可复现示例或 release 相关问题，请在 GitHub 仓库中打开 issue。
