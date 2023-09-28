# 9-26 
每个client内部数据的分布的差异对结果的影响。
1. client1:[1,2,3,4,5]; client2:[5,6,7,8,9]; client3[0,2,4,6,8]; client4[1,3,5,7,9] 与2 4*[1,2,3,4,5] 。 1 很快过拟合。2的效果比1好
# 8-29
every globl round 继续iter
# 8-27 
1.[local_train]  loacl round 作为 epoch, 每个epoch仅训练一次，将dataloader进行iter化，数据不shuffle。循环global_round时继续在dataloader里迭代
2. fednag+VGG16; fednag+linear; fedavg+VGG16; fedavg+linear
# fedml

main('linear',0.01,0.7,True,100,64,4,64)

python fednag.py  --model VGG16 --learning_rate 0.001 --momentum 0.7 --nesterov --num_rounds 1 --local_round 32 --num_clients 4 --batch_size 64

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_type", choices=['linear', 'other_model'], required=True, help="Specify the model type")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.7, help="Momentum")
    parser.add_argument("--nesterov", action="store_true", help="Enable Nesterov acceleration")
    parser.add_argument("--num_rounds", type=int, default=25, help="Number of training rounds")
    parser.add_argument("--local_round", type=int, default=5, help="Number of local training rounds")
    parser.add_argument("--num_clients", type=int, default=4, help="Number of clients")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

    args = parser.parse_args()
    main(args.model_type,args.learning_rate, args.momentum, args.nesterov ,args.num_rounds, args.local_round, args.num_clients,args.batch_size)



excel里面要有当前global 轮数，当前一共多少local iteration，average training loss， test loss， 还有accuracy
跑both fedavg和fednag 在cifar10和mnist数据库

local round = 40
momentum = 0.5
learning rate = 0.01
global round = 25
batch size = 64