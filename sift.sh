 ./exp/build/regann \
    dataset/sift/sift_vectors.fvecs \
    dataset/sift/sift_titles_clean.txt \
    dataset/sift/query.txt \
    10 100 \
    results/sift/ann.txt \
    30 ann \
    pq_m=8 load=results/sift/idx/sift gt=results/sift/gt.txt

 ./exp/build/regann \
     dataset/sift/sift_vectors.fvecs \
     dataset/sift/sift_titles_clean.txt \
     dataset/sift/query.txt \
     10 100 \
     results/sift/prefilter.txt \
     30 prefilter \
     gt=results/sift/gt.txt

 ./exp/build/regann \
     dataset/sift/sift_vectors.fvecs \
     dataset/sift/sift_titles_clean.txt \
     dataset/sift/query.txt \
     10 100 \
     results/sift/postfilter.txt \
     30 postfilter \
     gt=results/sift/gt.txt
