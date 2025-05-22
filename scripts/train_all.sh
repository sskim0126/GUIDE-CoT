cd goal_module
bash scripts/train_vis_sem.sh
cd ..

cd llm_module
bash scripts/trainval_g2p_all.sh
cd checkpoint
for d in "eth"; 
do
    cd LLM-$d-pixel-g2p
    for e in 0 1 2;
    do
        cp *.json epoch_$e
        cp spiece_model epoch_$e
    done
    cd ..
done
cd ..

