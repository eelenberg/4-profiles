# 4-Profile Readme

## 4profile.cpp

4-profile Counting.

Given an undirected graph, this program computes the frequencies of all subgraphs on 3 and 4 vertices (no automorphisms). A file counts_4_profilesLocal.txt is appended with input filename, edge sampling probability, 3-profile and 4-profile of the graph, and runtime. Network traffic is appended to netw_4_profLocal.txt similarly. Option (per_vertex) writes the local 3-profile and 4-profile, for each vertex the count of subgraphs including that vertex (including automorphisms). The algorithm assumes that each undirected edge appears exactly once in the graph input. If edges may appear more than once, this procedure 
will over count.

	./4profile --graph mygraph.txt --format tsv --sample_iter 10 --min_prob .5 --max_prob 1 --prob_step .1
	./4profile --graph mygraph.txt --format tsv --per_vertex local3


## getOnly2hop.cpp

Full 2-hop collection.

For each vertex, return all vertices 1 hop and 2 hops away from it. The algorithm assumes that each undirected edge appears exactly once in the graph input. If edges may appear more than once, this procedure will over count.

	./get2hopOnly --graph mygraph.txt --format tsv --list_file full2hoplist


## getOnlyHistogram.cpp

2-hop histogram calculation.

For each vertex, compute the 2-hop histogram (limited 2-hop information). The algorithm assumes that each undirected edge appears exactly once in the graph input. If edges may appear more than once, this procedure will over count.

	./getHistOnly --graph mygraph.txt --format tsv --sample_iter 10
