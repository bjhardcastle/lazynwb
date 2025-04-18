# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## Unreleased

<small>[Compare with latest](https://github.com/bjhardcastle/lazynwb/compare/v0.2.2...HEAD)</small>

### Added

- Add post commit hook for publishing ([f821852](https://github.com/bjhardcastle/lazynwb/commit/f8218526fc3ba8343d16fe1b089d41245047896f) by bjhardcastle).
- Add git config ([64f9fe9](https://github.com/bjhardcastle/lazynwb/commit/64f9fe933c54cf7d4a702f9b16b13bad48c594e2) by bjhardcastle).
- Add `get_timeseries()` docstring ([a3f4f6e](https://github.com/bjhardcastle/lazynwb/commit/a3f4f6ee1f140fc829fd39e6a1a572e14551ce16) by bjhardcastle).
- Add `get_df()` docstring ([e7a64e5](https://github.com/bjhardcastle/lazynwb/commit/e7a64e57a7475bfa820da7ca5302d9e67266f5ba) by bjhardcastle).

### Fixed

- Fix sorting df ([3a8dee6](https://github.com/bjhardcastle/lazynwb/commit/3a8dee65ad18d06fee890a88a20d15dedfca9c3c) by bjhardcastle).

### Removed

- Remove pdm lock ([59dd31d](https://github.com/bjhardcastle/lazynwb/commit/59dd31d335c6e24e4246a467ab1d4f3cfcc0ff10) by bjhardcastle).

<!-- insertion marker -->
## [v0.2.2](https://github.com/bjhardcastle/lazynwb/releases/tag/v0.2.2) - 2025-04-11

<small>[Compare with first commit](https://github.com/bjhardcastle/lazynwb/compare/c1551aa173dc2a94aeeecf50dd0324760720848b...v0.2.2)</small>

### Added

- Add `include_column_names` parameter to `get_df()` ([0856275](https://github.com/bjhardcastle/lazynwb/commit/0856275bc95d34d926b73e669710417b1e8de52f) by bjhardcastle).
- Add diagram ([03d1cf5](https://github.com/bjhardcastle/lazynwb/commit/03d1cf5cead1cb702a08a1eac924d40737ce2673) by bjhardcastle).
- Add `as_polars` option to metadata_df ([0a50489](https://github.com/bjhardcastle/lazynwb/commit/0a50489888db90b14f5c27a7dfe516880d62580a) by bjhardcastle).
- Add spike times functions WIP ([f0d305a](https://github.com/bjhardcastle/lazynwb/commit/f0d305aa9304e1fe64c52a4daecd7cbe0fb715a4) by bjhardcastle).
- Add options to suppress errors in `get_df()` ([8aea8ad](https://github.com/bjhardcastle/lazynwb/commit/8aea8ad0d62a989f2f24ddf0a11e1cc4c92f87cb) by bjhardcastle).
- Add missing return ([c369996](https://github.com/bjhardcastle/lazynwb/commit/c369996490ce40ca295c3e3bfed530785b654545) by bjhardcastle).
- Add metadata df speed test ([bbfed93](https://github.com/bjhardcastle/lazynwb/commit/bbfed935583db56842bf26a6a28a6a31ea81c38e) by bjhardcastle).
- Add dev dependencies for uv ([d8bf0c0](https://github.com/bjhardcastle/lazynwb/commit/d8bf0c064a98421da871c9a3803a19ef00ec5a6d) by bjhardcastle).
- Add anon S3 option to open() ([145646d](https://github.com/bjhardcastle/lazynwb/commit/145646df3e2dd76634a9a12ea75fef1e89ce0e3f) by bjhardcastle).
- Add standalone data interpreter fn ([ec0fee5](https://github.com/bjhardcastle/lazynwb/commit/ec0fee5c08994fa58ffcd65aee7f34b3cbe45548) by bjhardcastle).
- Add nwb_path to metadata df ([e2a8957](https://github.com/bjhardcastle/lazynwb/commit/e2a8957cbc25a3a4809e5fa40cd14aa63173b9a9) by bjhardcastle).
- Add reprs ([07e2b60](https://github.com/bjhardcastle/lazynwb/commit/07e2b60c68fb76fde856d536d1b04797520374df) by bjhardcastle).
- Add a describe() method ([6249ec8](https://github.com/bjhardcastle/lazynwb/commit/6249ec8335542da7adaa253b4de5e3460ec97cf9) by bjhardcastle).
- Add method for getting timeseries ([648cf89](https://github.com/bjhardcastle/lazynwb/commit/648cf8923d92ce41993dbe98a1d1b74d370f9756) by bjhardcastle).
- Add function and dataclass for getting timeseries ([f589367](https://github.com/bjhardcastle/lazynwb/commit/f58936790d9fc8f48c8601a745df52fad7a8e3b7) by bjhardcastle).
- Add progress bar for metadata df ([b9f39aa](https://github.com/bjhardcastle/lazynwb/commit/b9f39aa522009f0d516c8330526e4582ee56295c) by bjhardcastle).
- Add identifier property as alternative to session_id ([dfe8238](https://github.com/bjhardcastle/lazynwb/commit/dfe8238a14601ccd8823453b8832bbd740491fce) by bjhardcastle).
- Add tables to NWB class ([888d463](https://github.com/bjhardcastle/lazynwb/commit/888d463a86590df6caec16b54bd9235ba3856a65) by bjhardcastle).
- Add metadata df function ([1612f4a](https://github.com/bjhardcastle/lazynwb/commit/1612f4ab842450a38aad2e61a0fd4fc78c716b3c) by bjhardcastle).
- Add to_dict methods ([31862b9](https://github.com/bjhardcastle/lazynwb/commit/31862b9ad9e5fd988ee857e573bf5298c3394355) by bjhardcastle).
- Add function for merging array columns ([2a8b554](https://github.com/bjhardcastle/lazynwb/commit/2a8b554cdfa8ab860d2b7f96f6ff2dcd9ba92a50) by bjhardcastle).
- Add mods for sharing ([7956494](https://github.com/bjhardcastle/lazynwb/commit/79564947d9874158b6f29b1c7dca4050f35a894e) by bjhardcastle).
- Add object reference script ([4a4a0c9](https://github.com/bjhardcastle/lazynwb/commit/4a4a0c90342bca06dd9199cfb360af14f39a6a2f) by bjhardcastle).
- Add speed test comparing h5py with remfile ([919073f](https://github.com/bjhardcastle/lazynwb/commit/919073f857de87232ac3906a42622d663473e958) by bjhardcastle).
- Add storage options and fix access via http ([79a14a4](https://github.com/bjhardcastle/lazynwb/commit/79a14a47ebb1ceade86781674e5f40501a5b8ac5) by bjhardcastle).
- Add mutex lock and note on slow code ([76020d4](https://github.com/bjhardcastle/lazynwb/commit/76020d4da2a0ad862e6e350f3c2fee025d318643) by bjhardcastle).
- Add plots, scripts, data ([6b2e4d7](https://github.com/bjhardcastle/lazynwb/commit/6b2e4d747662e15487a6e2be0695fd63a409fe8c) by bjhardcastle).
- Add IBL helper ([4a5aba7](https://github.com/bjhardcastle/lazynwb/commit/4a5aba72a01e5482a355ab10e38cbb2cddbcc3fe) by bjhardcastle).
- Add dask ([059c3cf](https://github.com/bjhardcastle/lazynwb/commit/059c3cf98b22352f7e57ff3d93af8e90f4e59107) by bjhardcastle).
- Add filtered IBL assets and use context manager ([6d74c2c](https://github.com/bjhardcastle/lazynwb/commit/6d74c2c4834e27324b67d935c71925370d8437a9) by bjhardcastle).
- Add context manager to LazyNWB ([f5e25b3](https://github.com/bjhardcastle/lazynwb/commit/f5e25b32507637c346a25b1110d81d9a5adaa2e3) by bjhardcastle).
- Add notes about slow hdf5 access in README.md ([0d14197](https://github.com/bjhardcastle/lazynwb/commit/0d14197c903454bee56dbbddf29076abc5872ce8) by bjhardcastle).
- Add num_units field to Result class and update calculations ([cd2635e](https://github.com/bjhardcastle/lazynwb/commit/cd2635e1847e8f482f71114b52b7dcbab578a55f) by bjhardcastle).
- Add is_single_shank and is_v1_probe attributes to Result class ([3df445c](https://github.com/bjhardcastle/lazynwb/commit/3df445c9ed4d7a9d78ade0a6201348558e8d2835) by bjhardcastle).
- Add brain_region, most_common_area, debug ([3211f8a](https://github.com/bjhardcastle/lazynwb/commit/3211f8a8de1582cddfeae6b74abb606ed0d8115f) by bjhardcastle).
- Add dandi client fn and rename module ([d46709c](https://github.com/bjhardcastle/lazynwb/commit/d46709c2b20fffa7cc6d57d73d7bd0e26df05b7e) by bjhardcastle).
- Add script to fetch unit yield ([72c1265](https://github.com/bjhardcastle/lazynwb/commit/72c1265edba67a3fd7029a77139ab9a97a53da46) by bjhardcastle).
- Add dandiset functions ([d95fd07](https://github.com/bjhardcastle/lazynwb/commit/d95fd0738883b0f4e7fc7373fc375a5d03b78baf) by bjhardcastle).

### Fixed

- Fix types ([79baed7](https://github.com/bjhardcastle/lazynwb/commit/79baed7b29dc74d9dd187bb3e68f68755b554260) by bjhardcastle).
- Fix task command ([8299bf0](https://github.com/bjhardcastle/lazynwb/commit/8299bf01e6e5c44f8671c283e26558914f788dc0) by bjhardcastle).
- Fix type ([d4da20e](https://github.com/bjhardcastle/lazynwb/commit/d4da20ed18409e194c6d16766f733e6f89ee24d4) by bjhardcastle).
- Fix varname ([59918a9](https://github.com/bjhardcastle/lazynwb/commit/59918a93d96ccb1845f46cc248b87427ef570900) by bjhardcastle).
- Fix include column filtering ([c2d2603](https://github.com/bjhardcastle/lazynwb/commit/c2d2603e5adc0ee620c70fa0b1a8f7c5aa092540) by bjhardcastle).
- Fix filtering of intervals table ([e5eec96](https://github.com/bjhardcastle/lazynwb/commit/e5eec9616d4873ad78c24b27a2044f2dd0b1edc0) by bjhardcastle).
- Fix getting subject metadata ([ac5b489](https://github.com/bjhardcastle/lazynwb/commit/ac5b4893a3c5da86cad82a403f5eb509ccd60c8f) by bjhardcastle).
- Fix variable name ([b506648](https://github.com/bjhardcastle/lazynwb/commit/b50664837d1b293d4ea091af76e39f7118f45eca) by bjhardcastle).
- Fix path to function ([afc187f](https://github.com/bjhardcastle/lazynwb/commit/afc187ff1f43d466b11100ab94025636c935840d) by bjhardcastle).
- Fix progress bar for spikes in intervals ([a13888d](https://github.com/bjhardcastle/lazynwb/commit/a13888dff1ea91ed720adba572fd3c21b5423462) by bjhardcastle).
- Fix getting obs intervals ([f680f5e](https://github.com/bjhardcastle/lazynwb/commit/f680f5e71873b9e1a55d1e570ae7f8194e9be59c) by bjhardcastle).
- Fix `is_observed()` ([fd43944](https://github.com/bjhardcastle/lazynwb/commit/fd439449ab08831369325581b949a6f7d8da1b36) by bjhardcastle).
- Fix `Timeseries.timestamps` ([c208fda](https://github.com/bjhardcastle/lazynwb/commit/c208fda2b5ee203eb9088bd2eaec04eee5b224b0) by bjhardcastle).
- Fix getting timeseries ([4ab5de1](https://github.com/bjhardcastle/lazynwb/commit/4ab5de17c2911c7b9a90cdd7135af060982122f8) by bjhardcastle).
- Fix exception handling ([6000bdd](https://github.com/bjhardcastle/lazynwb/commit/6000bddbb374136590d0749ffcf36157805118ea) by bjhardcastle).
- Fix `get_df('units')` with DR datacube #2 ([05b9816](https://github.com/bjhardcastle/lazynwb/commit/05b9816d0f811a8435c9b55ae209e7007bc39ccd) by bjhardcastle).
- Fix deprecated config ([994bf64](https://github.com/bjhardcastle/lazynwb/commit/994bf6469badfe0d6099f6a0b1e1e473c2493687) by bjhardcastle).
- Fix list[str] bug ([958d830](https://github.com/bjhardcastle/lazynwb/commit/958d83088b1bf943c93a5fe9d650fa25e4f94ca3) by bjhardcastle).
- Fix deadlock ([8e7179f](https://github.com/bjhardcastle/lazynwb/commit/8e7179f851907c1b6aa044d8805861b3f0c2bb3b) by bjhardcastle).
- Fix get method ([9bd5cd4](https://github.com/bjhardcastle/lazynwb/commit/9bd5cd44a4bc3d894227afc00a2bc89cd619a8ba) by bjhardcastle).
- Fix reprs ([efd0286](https://github.com/bjhardcastle/lazynwb/commit/efd02869b022ed7d2cf4244b44cb318a4a57ecf6) by bjhardcastle).
- Fix keywords display in repr ([433cf11](https://github.com/bjhardcastle/lazynwb/commit/433cf1191453f7bd5cd8e3431b7297960f2be172) by bjhardcastle).
- Fix getting TimeSeries for hdf5 files ([37a2422](https://github.com/bjhardcastle/lazynwb/commit/37a242214bfc3007259ef186e66804e98840a51d) by bjhardcastle).
- Fix timeseries dict keys ([06d650e](https://github.com/bjhardcastle/lazynwb/commit/06d650e11c3efd13706e24c3961f870e0eb88dde) by bjhardcastle).
- Fix method for getting timeseries ([7db24d4](https://github.com/bjhardcastle/lazynwb/commit/7db24d4852b9a305f971ac3bb5e5b1fd74ec2124) by bjhardcastle).
- Fix keeping multi-dimensional columns as np arrays ([b160375](https://github.com/bjhardcastle/lazynwb/commit/b16037514e5b959507b4a32829facfce803144f7) by bjhardcastle).
- Fix error message ([ba2315f](https://github.com/bjhardcastle/lazynwb/commit/ba2315f72b4bbcd3329ce39d984ce37f69958b9b) by bjhardcastle).
- Fix accessing identifier ([e52fe01](https://github.com/bjhardcastle/lazynwb/commit/e52fe01c98b800f8bfa1b8d154e8daa3b02b8a7a) by bjhardcastle).
- Fix variable name and type ([395d5e7](https://github.com/bjhardcastle/lazynwb/commit/395d5e7f182d30afb7b3e3beef7dc7cc7d6119d5) by bjhardcastle).
- Fix geting indexed column data ([046792f](https://github.com/bjhardcastle/lazynwb/commit/046792f03ad1ba5a5eaa7cccd377605ae5eef0b3) by bjhardcastle).
- Fix identifier in table dataframe ([60a7882](https://github.com/bjhardcastle/lazynwb/commit/60a788265f15675645b1a78b02fd140b25c3f7da) by bjhardcastle).
- Fix loading multi-dim arrays into dataframe ([9c1b39e](https://github.com/bjhardcastle/lazynwb/commit/9c1b39e1b00d533279da83cf0181e663bfc0749a) by bjhardcastle).
- Fix indexing for zarr arrays ([78d66cb](https://github.com/bjhardcastle/lazynwb/commit/78d66cb7d3aa7dc7f3e680b5a39a993ef8ef5f73) by bjhardcastle).
- Fix <3.11 compatibility ([ac3092d](https://github.com/bjhardcastle/lazynwb/commit/ac3092db94c0e33dd64fd1d7c9c8366548ac431d) by bjhardcastle).
- Fix getting dataframes ([2c3d7ee](https://github.com/bjhardcastle/lazynwb/commit/2c3d7ee444a87eaa5394b45a7ac9758c33862497) by bjhardcastle).
- Fix units -> good units ([f3cb058](https://github.com/bjhardcastle/lazynwb/commit/f3cb058f47a1b80a6b2be0b6f2699bff4b464109) by bjhardcastle).
- Fix num_units calculation in get_unit_yield_over_time.py ([7df4d0d](https://github.com/bjhardcastle/lazynwb/commit/7df4d0db68e15c0d93a802963e7d280573c6dced) by bjhardcastle).
- Fix follow_redirects parameter in get_lazynwb_from_dandiset_asset function ([8503a2c](https://github.com/bjhardcastle/lazynwb/commit/8503a2cb0e8d09669ab56c93c9f2087f02b4f530) by bjhardcastle).
- Fix import ([c6e7ec6](https://github.com/bjhardcastle/lazynwb/commit/c6e7ec6bc2a1a4131e428987472cd1aac416c5cf) by bjhardcastle).
- Fix 3.9 compatibility ([e2a7af1](https://github.com/bjhardcastle/lazynwb/commit/e2a7af184ebb95fc1ccba72cbb0d433a7baa3fab) by bjhardcastle).

### Removed

- Remove LazyComponent for performance ([287458c](https://github.com/bjhardcastle/lazynwb/commit/287458c4e3a62fd75781772f024abd4a958d9be0) by bjhardcastle).
- Remove warning ([0747093](https://github.com/bjhardcastle/lazynwb/commit/0747093bc2d37c239b5cb1168109d7249262bcee) by bjhardcastle).
- Remove anon s3 access Causing permission error on private bucket ([6a16529](https://github.com/bjhardcastle/lazynwb/commit/6a1652954806596e59e1b1185a8c985442c30157) by bjhardcastle).
- Remove old function ([b30bfd7](https://github.com/bjhardcastle/lazynwb/commit/b30bfd73ba061055ab6946ef8b07ee8ac228c1a4) by bjhardcastle).
- Remove polars experiments ([b48ff10](https://github.com/bjhardcastle/lazynwb/commit/b48ff1071a0ad7322de281349cc2a0fdd347024c) by bjhardcastle).
- Remove redundant data access ([57e1385](https://github.com/bjhardcastle/lazynwb/commit/57e13858a83f82bf17634d02f7513b62d4139be5) by bjhardcastle).
- Remove strip on dandi query parameters ([65f0f76](https://github.com/bjhardcastle/lazynwb/commit/65f0f76feb9821f1468f66623b7a2167a4dc0b40) by bjhardcastle).

