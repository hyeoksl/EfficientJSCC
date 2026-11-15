uv run train.py --config config.yaml --experiment_name ConvNeXt_wide_nogroup --model.type DeepJSCC_main --model.dims=[64,128,256,512]
uv run train.py --config config.yaml --experiment_name ConvNeXt_wide_itgroup --model.type DeepJSCC_main --model.dims=[64,128,256,512] --model.itgroup=[1,2,4,8]
uv run train.py --config config.yaml --experiment_name ConvNeXt_wide_resgroup --model.type DeepJSCC_main --model.dims=[64,128,256,512] --model.resgroup=[1,2,4,8]
uv run train.py --config config.yaml --experiment_name ConvNeXt_dense_nogroup --model.type DeepJSCC_main --model.dims=[280,280,280,280]
uv run train.py --config config.yaml --experiment_name ConvNeXt_wide_allgroup --model.type DeepJSCC_main --model.dims=[64,128,256,512] --model.resgroup=[1,2,4,8] --model.itgroup=[1,2,4,8]

uv run train.py --config config.yaml --experiment_name ConvNeXt_wide_itgroup_unshuffle --model.type DeepJSCC_main --model.dims=[64,128,256,512] --model.itgroup=[1,2,4,8] --model.shuffle False
uv run train.py --config config.yaml --experiment_name ConvNeXt_wide_resgroup_unshuffle --model.type DeepJSCC_main --model.dims=[64,128,256,512] --model.resgroup=[1,2,4,8] --model.shuffle False
uv run train.py --config config.yaml --experiment_name ConvNeXt_dense_nogroup_unshuffle --model.type DeepJSCC_main --model.dims=[280,280,280,280] --model.shuffle False
uv run train.py --config config.yaml --experiment_name ConvNeXt_wide_allgroup_unshuffle --model.type DeepJSCC_main --model.dims=[64,128,256,512] --model.resgroup=[1,2,4,8] --model.itgroup=[1,2,4,8] --model.shuffle False

uv run train.py --config config.yaml --experiment_name ConvNeXt_wide_itgroup_noall --model.type DeepJSCC_main --model.dims=[64,128,256,512] --model.itgroup=[1,2,4,8] --model.normgroup True --model.shuffle False
uv run train.py --config config.yaml --experiment_name ConvNeXt_wide_resgroup_noall --model.type DeepJSCC_main --model.dims=[64,128,256,512] --model.resgroup=[1,2,4,8] --model.normgroup True --model.shuffle False
uv run train.py --config config.yaml --experiment_name ConvNeXt_dense_nogroup_noall --model.type DeepJSCC_main --model.dims=[280,280,280,280] --model.normgroup True --model.shuffle False
uv run train.py --config config.yaml --experiment_name ConvNeXt_wide_allgroup_noall --model.type DeepJSCC_main --model.dims=[64,128,256,512] --model.resgroup=[1,2,4,8] --model.itgroup=[1,2,4,8] --model.normgroup True --model.shuffle False

uv run train.py --config config.yaml --experiment_name ConvNeXt_wide_itgroup_nogroup --model.type DeepJSCC_main --model.dims=[64,128,256,512] --model.itgroup=[1,2,4,8] --model.normgroup True
uv run train.py --config config.yaml --experiment_name ConvNeXt_wide_resgroup_nogroup --model.type DeepJSCC_main --model.dims=[64,128,256,512] --model.resgroup=[1,2,4,8] --model.normgroup True
uv run train.py --config config.yaml --experiment_name ConvNeXt_dense_nogroup_nogroup --model.type DeepJSCC_main --model.dims=[280,280,280,280] --model.normgroup True
uv run train.py --config config.yaml --experiment_name ConvNeXt_wide_allgroup_nogroup --model.type DeepJSCC_main --model.dims=[64,128,256,512] --model.resgroup=[1,2,4,8] --model.itgroup=[1,2,4,8] --model.normgroup True