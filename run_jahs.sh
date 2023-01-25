for dataset in cifar10 fashion_mnist colorectal_histology
do
    echo $dataset
    python collect_jahs.py --dataset $dataset
    python analyze_jahs.py --dataset $dataset
done
