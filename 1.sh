#./exp/build/regann \
#    dataset/siftsmall/siftsmall_vectors.fvecs \
#    dataset/siftsmall/siftsmall_titles_clean.txt \
#    dataset/siftsmall/query.txt \
#    10 100 \
#    results/siftsmall/gt.txt \
#    30 groundtruth

#./exp/build/regann \
#    dataset/siftsmall/siftsmall_vectors.fvecs \
#    dataset/siftsmall/siftsmall_titles_clean.txt \
#    dataset/siftsmall/query.txt \
#    10 100 \
#    results/siftsmall/ann.txt \
#    30 ann \
#    pq_m=8 save=results/siftsmall/idx/siftsmall gt=results/siftsmall/gt.txt

# ./exp/build/regann \
#     dataset/siftsmall/siftsmall_vectors.fvecs \
#     dataset/siftsmall/siftsmall_titles_clean.txt \
#     dataset/siftsmall/query.txt \
#     10 100 \
#     results/siftsmall/ann.txt \
#     30 prefilter \
#     pq_m=8 save=results/siftsmall/idx/siftsmall gt=results/siftsmall/gt.txt

./exp/build/regann \
    dataset/siftsmall/siftsmall_vectors.fvecs \
    dataset/siftsmall/siftsmall_titles_clean.txt \
    dataset/siftsmall/query.txt \
    10 100 \
    results/siftsmall/ann.txt \
    30 postfilter \
    oversample=500 \
    pq_m=8 save=results/siftsmall/idx/siftsmall gt=results/siftsmall/gt.txt