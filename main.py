import numpy as np
import cma.optimizer.meta_cmaes_elitist as meta_e
import cma.optimizer.cmaes_elitist as cma_el
import cma.objective_function.continuous as cont
import cma.util.sampler as sampler
import cma.util.log as log

def sharpridge_optimization_run(dim, meta_f, inner_f, verbose=False, seed=42): 
    # 再現性のため固定シードの乱数生成器を使う
    rng = np.random.default_rng(seed)
    # 初期値の設定
    init_m = rng.standard_normal(dim)
    init_sigma = 1 

    # サンプラーの準備 (今回はエリート主義なので評価は1回)
    meta_samp = sampler.Sampler(meta_f, 1)
    inner_samp = sampler.Sampler(inner_f, 1)

    # CMAES_elitistのインスタンスを作成
    opt = meta_e.Meta_CMAES_elitist(
        d=dim,
        meta_sampler=meta_samp,
        inner_sampler=inner_samp,
        m=init_m,
        sigma=init_sigma,
        min_problem=meta_f.minimization_problem
    )
    

    logger = log.DataLogger("output/log.csv")
                    
    # 最適化を実行
    return opt.run(logger, verbose=verbose)
    # return opt.run(verbose=verbose)

def main():
    # ------------------------------
    # setting
    # ------------------------------
    dim = 20              # total number of dimensions
    max_eval = dim * 1e7    # maximum number of evaluations
    d_coeff = 1e6
    alpha = 1/8

    # ------------------------------
    # select objective function
    # ------------------------------

    avg_eval = 0
    correct = 0

    meta_f = cont.SharpRidge(dim, d_coeff = d_coeff, alpha = alpha, max_eval = max_eval)
    inner_f = cont.SharpRidge(dim, d_coeff = d_coeff, alpha = alpha, max_eval = 10000)
    result = sharpridge_optimization_run(dim, meta_f, inner_f, verbose=False, seed=42)
    print("Number of evaluations: {}".format(result[0]))
    print("Best evaluation value: {}".format(result[1]))
    print("Is the best evaluation value better than the target evaluation value?: {}".format(result[2]))
    if result[2]:
        avg_eval += result[0]
        correct += 1

if __name__ == "__main__":
    main()
