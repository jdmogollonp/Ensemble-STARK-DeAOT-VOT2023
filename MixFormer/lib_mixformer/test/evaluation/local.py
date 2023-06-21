from lib_mixformer.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/mnt/pixstor/data/grzc7/MixFormer/data/got10k_lmdb'
    settings.got10k_path = '/mnt/pixstor/data/grzc7/MixFormer/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/mnt/pixstor/data/grzc7/MixFormer/data/lasot_lmdb'
    settings.lasot_path = '/mnt/pixstor/data/grzc7/MixFormer/data/lasot'
    settings.network_path = '/mnt/pixstor/data/grzc7/MixFormer/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/mnt/pixstor/data/grzc7/MixFormer/data/nfs'
    settings.otb_path = '/mnt/pixstor/data/grzc7/MixFormer/data/OTB2015'
    settings.prj_dir = '/mnt/pixstor/data/grzc7/MixFormer'
    settings.result_plot_path = '/mnt/pixstor/data/grzc7/MixFormer/test/result_plots'
    settings.results_path = '/mnt/pixstor/data/grzc7/MixFormer/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/mnt/pixstor/data/grzc7/MixFormer'
    settings.segmentation_path = '/mnt/pixstor/data/grzc7/MixFormer/test/segmentation_results'
    settings.tc128_path = '/mnt/pixstor/data/grzc7/MixFormer/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/mnt/pixstor/data/grzc7/MixFormer/data/trackingNet'
    settings.uav_path = '/mnt/pixstor/data/grzc7/MixFormer/data/UAV123'
    settings.vot_path = '/mnt/pixstor/data/grzc7/MixFormer/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

