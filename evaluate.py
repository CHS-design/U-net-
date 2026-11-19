"""evaluate.py — 测试集评估入口（阶段 6 实现）

职责：
    1. 从 checkpoints/{RUN_NAME}/best.pt 加载 state_dict
    2. model.eval() 后在 test 集上统计 Dice / IoU / 像素准确率
    3. test 集只在最后用一次；训练期选优只用 val
    4. 输出总指标，必要时按类别或样本打印

状态：空壳，等阶段 6 亲手实现。
"""
