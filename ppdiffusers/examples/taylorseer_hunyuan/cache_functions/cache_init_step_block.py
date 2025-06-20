def cache_init_step_block(num_steps, model_kwargs=None):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1]={}
    cache_index[-1]={}
    cache_index['layer_index']={}
    cache['hidden'] = {}
    cache['firstblock_hidden']= {}
    # cache_dic['attn_map'] = {}
    # cache_dic['attn_map'][-1] = {}
    # cache_dic['attn_map'][-1]['double_stream'] = {}
    # cache_dic['attn_map'][-1]['single_stream'] = {}

    # cache_dic['k-norm'] = {}
    # cache_dic['k-norm'][-1] = {}
    # cache_dic['k-norm'][-1]['double_stream'] = {}
    # cache_dic['k-norm'][-1]['single_stream'] = {}

    # cache_dic['v-norm'] = {}
    # cache_dic['v-norm'][-1] = {}
    # cache_dic['v-norm'][-1]['double_stream'] = {}
    # cache_dic['v-norm'][-1]['single_stream'] = {}

    # cache_dic['cross_attn_map'] = {}
    # cache_dic['cross_attn_map'][-1] = {}
    # cache[-1]['double_stream']={}
    # cache[-1]['single_stream']={}
    cache_dic['cache_counter'] = 0

    cache_dic['taylor_cache'] = False
    cache_dic['duca']  = False
    cache_dic['test_FLOPs'] = False

    mode = 'Taylor'
    if mode == 'original':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa'
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = 1
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 1
        
    elif mode == 'ToCa':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.10
        cache_dic['fresh_threshold'] = 5
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 1
        cache_dic['duca'] = False

    elif mode == 'DuCa':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.10
        cache_dic['fresh_threshold'] = 5
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 1
        cache_dic['duca'] = True

    elif mode == 'Taylor':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = 2
        cache_dic['max_order'] = 1
        cache_dic['firstblock_max_order'] = 1
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['taylor_cache'] = True
        cache_dic['first_enhance'] = 1

    current = {}
    current['num_steps'] = num_steps
    current['activated_steps'] = [0]
    current['block_activated_steps'] = [0]

    return cache_dic, current
