/*
getOnly2hop.cpp Compute full 2-hop connectivity information
Distributed Estimation of Graph 4-profiles
Ethan R. Elenberg, Karthikeyan Shanmugam, Michael Borokhovich, Alexandros G. Dimakis
http://github.com/eelenberg/4-profiles
*/

/*  
 * Copyright (c) 2009 Carnegie Mellon University. 
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://www.graphlab.ml.cmu.edu
 *
 */


#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp> 
#include <graphlab.hpp>
#include <graphlab/ui/metrics_server.hpp>
#include <graphlab/util/hopscotch_set.hpp>
#include <graphlab/macros_def.hpp>

//using namespace boost::multiprecision;

 
// Radix sort implementation from https://github.com/gorset/radix
// Thanks to Erik Gorset
//
/*
Copyright 2011 Erik Gorset. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list
of conditions and the following disclaimer in the documentation and/or other materials
provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Erik Gorset ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Erik Gorset OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Erik Gorset.
*/

//global variables
double sample_prob_keep = 1;
double min_prob = 1;
double max_prob = 1;
double prob_step = 0.5;
size_t total_edges = 0;
int sample_iter = 1;

void radix_sort(graphlab::vertex_id_type *array, int offset, int end, int shift) {
    int x, y;
    graphlab::vertex_id_type value, temp;
    int last[256] = { 0 }, pointer[256];

    for (x=offset; x<end; ++x) {
        ++last[(array[x] >> shift) & 0xFF];
    }

    last[0] += offset;
    pointer[0] = offset;
    for (x=1; x<256; ++x) {
        pointer[x] = last[x-1];
        last[x] += last[x-1];
    }

    for (x=0; x<256; ++x) {
        while (pointer[x] != last[x]) {
            value = array[pointer[x]];
            y = (value >> shift) & 0xFF;
            while (x != y) {
                temp = array[pointer[y]];
                array[pointer[y]++] = value;
                value = temp;
                y = (value >> shift) & 0xFF;
            }
            array[pointer[x]++] = value;
        }
    }

    if (shift > 0) {
        shift -= 8;
        for (x=0; x<256; ++x) {
            temp = x > 0 ? pointer[x] - pointer[x-1] : pointer[0] - offset;
            if (temp > 64) {
                radix_sort(array, pointer[x] - temp, pointer[x], shift);
            } else if (temp > 1) {
                std::sort(array + (pointer[x] - temp), array + pointer[x]);
                //insertion_sort(array, pointer[x] - temp, pointer[x]);
            }
        }
    }
}

size_t HASH_THRESHOLD = 64;
// size_t HASH_THRESHOLD2 = 45000; //for nbh_count

// We on each vertex, either a vector of sorted VIDs
// or a hash set (cuckoo hash) of VIDs.
// If the number of elements is greater than HASH_THRESHOLD,
// the hash set is used. Otherwise the vector is used.
struct vid_vector{
  std::vector<graphlab::vertex_id_type> vid_vec;
  graphlab::hopscotch_set<graphlab::vertex_id_type> *cset;
  vid_vector(): cset(NULL) { }
  vid_vector(const vid_vector& v):cset(NULL) {
    (*this) = v;
  }

  vid_vector& operator=(const vid_vector& v) {
    if (this == &v) return *this;
    vid_vec = v.vid_vec;
    if (v.cset != NULL) {
      // allocate the cuckoo set if the other side is using a cuckoo set
      // or clear if I alrady have one
      if (cset == NULL) {
        cset = new graphlab::hopscotch_set<graphlab::vertex_id_type>(HASH_THRESHOLD);
      }
      else {
        cset->clear();
      }
      (*cset) = *(v.cset);
    }
    else {
      // if the other side is not using a cuckoo set, lets not use a cuckoo set
      // either
      if (cset != NULL) {
        delete cset;
        cset = NULL;
      }
    }
    return *this;
  }

  ~vid_vector() {
    if (cset != NULL) delete cset;
  }

  // assigns a vector of vertex IDs to this storage.
  // this function will clear the contents of the vid_vector
  // and reconstruct it.
  // If the assigned values has length >= HASH_THRESHOLD,
  // we will allocate a cuckoo set to store it. Otherwise,
  // we just store a sorted vector
  void assign(const std::vector<graphlab::vertex_id_type>& vec) {
    clear();
    if (vec.size() >= HASH_THRESHOLD) {
        // move to cset
        cset = new graphlab::hopscotch_set<graphlab::vertex_id_type>(HASH_THRESHOLD);
        foreach (graphlab::vertex_id_type v, vec) {
          cset->insert(v);
        }
    }
    else {
      vid_vec = vec;
      if (vid_vec.size() > 64) {
        radix_sort(&(vid_vec[0]), 0, vid_vec.size(), 24);
      }
      else {
        std::sort(vid_vec.begin(), vid_vec.end());
      }
      //do not maintain unique elements?
      std::vector<graphlab::vertex_id_type>::iterator new_end = std::unique(vid_vec.begin(),
                                               vid_vec.end());
      vid_vec.erase(new_end, vid_vec.end());
    }
  }

  
  void save(graphlab::oarchive& oarc) const {
    oarc << (cset != NULL);
    if (cset == NULL) oarc << vid_vec;
    else oarc << (*cset);
  }


  void clear() {
    vid_vec.clear();
    if (cset != NULL) {
      delete cset;
      cset = NULL;
    }
  }

  size_t size() const {
    return cset == NULL ? vid_vec.size() : cset->size();
  }

  void load(graphlab::iarchive& iarc) {
    clear();
    bool hascset;
    iarc >> hascset;
    if (!hascset) iarc >> vid_vec;
    else {
      cset = new graphlab::hopscotch_set<graphlab::vertex_id_type>(HASH_THRESHOLD);
      iarc >> (*cset);
    }
  }
};

/*
  A simple counting iterator which can be used as an insert iterator.
  but only counts the number of elements inserted. Useful for
  use with counting the size of an intersection using std::set_intersection
*/
template <typename T>
struct counting_inserter {
  size_t* i;
  counting_inserter(size_t* i):i(i) { }
  counting_inserter& operator++() {
    ++(*i);
    return *this;
  }
  void operator++(int) {
    ++(*i);
  }

  struct empty_val {
    empty_val operator=(const T&) { return empty_val(); }
  };

  empty_val operator*() {
    return empty_val();
  }

  typedef empty_val reference;
};


/*
 * Each vertex maintains a list of all its neighbors.
 * and a final count for the number of triangles it is involved in
 */
struct vertex_data_type {
  //vertex_data_type(): num_triangles(0){ }
  // A list of all its neighbors
  vid_vector vid_set;
  std::vector<vid_vector> two_hop_set; //things that are at most 2 hops away
  // double num_moments;
  // double sum_counts;
  
  void save(graphlab::oarchive &oarc) const {
    oarc << vid_set << two_hop_set;
  }
  void load(graphlab::iarchive &iarc) {
    iarc >> vid_set >> two_hop_set;
  }
};



/*
 * Each edge is simply a counter of triangles
 *
 */
//typedef uint32_t edge_data_type;

struct edge_data_type {
  bool sample_indicator;
  void save(graphlab::oarchive &oarc) const {
    oarc << sample_indicator;
  }
  void load(graphlab::iarchive &iarc) {
    iarc >> sample_indicator;
  }
};


bool PER_VERTEX_COUNT = false;


/*
 * This is the gathering type which accumulates an array of
 * all neighboring vertices.
 * It is a simple wrapper around a vector with
 * an operator+= which simply performs a  +=
 */
struct set_union_gather {
  graphlab::vertex_id_type v;
  std::vector<graphlab::vertex_id_type> vid_vec;
  
  set_union_gather():v(-1) {
  }

  size_t size() const {
    if (v == (graphlab::vertex_id_type)-1) return vid_vec.size();
    else return 1;
  }
  /*
   * Combining with another collection of vertices.
   * Union it into the current set.
   */
  set_union_gather& operator+=(const set_union_gather& other) {
    if (size() == 0) {
      (*this) = other;
      return (*this);
    }
    else if (other.size() == 0) {
      return *this;
    }

    if (vid_vec.size() == 0) {
      vid_vec.push_back(v);
      v = (graphlab::vertex_id_type)(-1);
    }
    if (other.vid_vec.size() > 0) {
      size_t ct = vid_vec.size();
      vid_vec.resize(vid_vec.size() + other.vid_vec.size());
      for (size_t i = 0; i < other.vid_vec.size(); ++i) {
        vid_vec[ct + i] = other.vid_vec[i];
      }
    }
    else if (other.v != (graphlab::vertex_id_type)-1) {
      vid_vec.push_back(other.v);
    }
    return *this;
  }
  
  // serialize
  void save(graphlab::oarchive& oarc) const {
    oarc << bool(vid_vec.size() == 0);
    if (vid_vec.size() == 0) oarc << v;
    else oarc << vid_vec;
  }

  // deserialize
  void load(graphlab::iarchive& iarc) {
    bool novvec;
    v = (graphlab::vertex_id_type)(-1);
    vid_vec.clear();
    iarc >> novvec;
    if (novvec) iarc >> v;
    else iarc >> vid_vec;
  }
};


struct vector_push_gather {
  vid_vector n;
  std::vector<vid_vector> nbh_vec;
  
  //if n is empty the nbh vector is full, if not n has a single vid_vec
  size_t size() const {
    if (n.size() == 0) return nbh_vec.size();
    else return 1;
  }
  /*
   * Combining with another collection of vertices.
   * Union it into the current set.
   */
  vector_push_gather& operator+=(const vector_push_gather& other) {
    if (size() == 0) {
      (*this) = other;
      return (*this);
    }
    else if (other.size() == 0) {
      return *this;
    }

    //if nbh vector is empty, move n into it and clear n
    if (nbh_vec.size() == 0) {
      nbh_vec.push_back(n);
      n.clear();
    }
    //if nbh vector full, move it into this, else push n into this
    if (other.nbh_vec.size() > 0) {
      size_t ct = nbh_vec.size();
      nbh_vec.resize(nbh_vec.size() + other.nbh_vec.size());
      // std::cout << "OLD SIZE: " << ct << " NEW SIZE: " << vid_vec.size() << "\n";
      for (size_t i = 0; i < other.nbh_vec.size(); ++i) {
        nbh_vec[ct + i] = other.nbh_vec[i];
      }
    }
    else if (other.n.size() != 0) {
      nbh_vec.push_back(other.n);
    }
    return *this;
  }
  
  // serialize
  void save(graphlab::oarchive& oarc) const {
    oarc << bool(nbh_vec.empty());
    if (nbh_vec.empty()) oarc << n;
    else oarc << nbh_vec;
  }

  // deserialize
  void load(graphlab::iarchive& iarc) {
    bool novvec;
    std::vector<vid_vector> n;
    n.clear();
    nbh_vec.clear();
    iarc >> novvec;
    if (novvec) iarc >> n;
    else iarc >> nbh_vec;
  }
};



/*
 * Define the type of the graph
 */
typedef graphlab::distributed_graph<vertex_data_type,
                                    edge_data_type> graph_type;


// //move init outside constructor (must be declared after graph_type)
void init_vertex(graph_type::vertex_type& vertex) { 
     
     vertex.data().two_hop_set.clear();
     vertex.data().vid_set.clear();
}

void clear_2hop_data(graph_type::vertex_type& vertex) { 
     vertex.data().two_hop_set.clear();
}

  void sample_edge(graph_type::edge_type& edge) {
     if(graphlab::random::rand01() < sample_prob_keep)   
       edge.data().sample_indicator = 1;
     else
       edge.data().sample_indicator = 0;
  }


/*
 * Gather full 1-hop neighborhood
 */
class hop_count :
      public graphlab::ivertex_program<graph_type,
                                      set_union_gather>,
      /* I have no data. Just force it to POD */
      public graphlab::IS_POD_TYPE  {
public:
  bool do_not_scatter;

  // Gather on all edges
  edge_dir_type gather_edges(icontext_type& context,
                             const vertex_type& vertex) const {
    //sample edges here eventually, maybe by random edge.id() so consistent between the 2 engines?
    return graphlab::ALL_EDGES;
  } 

  /*
   * For each edge, figure out the ID of the "other" vertex
   * and accumulate a set of the neighborhood vertex IDs.
   */
  gather_type gather(icontext_type& context,
                     const vertex_type& vertex,
                     edge_type& edge) const {
    set_union_gather gather;
    if(edge.data().sample_indicator == 0){
       gather.v = -1; //default value
     }
     else{
      graphlab::vertex_id_type otherid = edge.target().id() == vertex.id() ?
                                       edge.source().id() : edge.target().id();

    // size_t other_nbrs = (edge.target().id() == vertex.id()) ?
    //     (edge.source().num_in_edges() + edge.source().num_out_edges()): 
    //     (edge.target().num_in_edges() + edge.target().num_out_edges());

    // size_t my_nbrs = vertex.num_in_edges() + vertex.num_out_edges();

    //if (PER_VERTEX_COUNT || (other_nbrs > my_nbrs) || (other_nbrs == my_nbrs && otherid > vertex.id())) {
    //if (PER_VERTEX_COUNT || otherid > vertex.id()) {
      // std::cout << "THIS ID: " << vertex.id() << " OTHER ID: " << otherid << "\n";
      gather.v = otherid; //will this work? what is v??
    //} 
   } 
   return gather;
  }

  /*
   * the gather result now contains the vertex IDs in the neighborhood.
   * store it on the vertex. 
   */
  void apply(icontext_type& context, vertex_type& vertex,
             const gather_type& neighborhood) {
   do_not_scatter = false;
   if (neighborhood.vid_vec.size() == 0) {
     // neighborhood set may be empty or has only 1 element
     vertex.data().vid_set.clear();
     if (neighborhood.v != (graphlab::vertex_id_type(-1))) {
       vertex.data().vid_set.vid_vec.push_back(neighborhood.v);
     }
   }
   else {
     vertex.data().vid_set.assign(neighborhood.vid_vec);
   }
   do_not_scatter = vertex.data().vid_set.size() == 0;
  } // end of apply

  /*
   * Scatter over all edges to compute the intersection.
   * I only need to touch each edge once, so if I scatter just on the
   * out edges, that is sufficient.
   */
   //change to no scatter
  edge_dir_type scatter_edges(icontext_type& context,
                              const vertex_type& vertex) const {
    // if (do_not_scatter) return graphlab::NO_EDGES;
    // else return graphlab::OUT_EDGES;
    return graphlab::NO_EDGES;
  }

 };


/* now another class for 2 hop neighborhood: for each edge, exchange 1-hop infos to get 2-hop info*/
//right now this lists everything at most 2 hops away
class two_hop_count :
public graphlab::ivertex_program<graph_type,
                                      vector_push_gather>,
      /* I have no data. Just force it to POD */
      public graphlab::IS_POD_TYPE  {
public:
  bool do_not_scatter;

  // Gather on all edges
  edge_dir_type gather_edges(icontext_type& context,
                             const vertex_type& vertex) const {
    //sample edges here eventually, maybe by random edge.id() so consistent between the 2 engines?
    return graphlab::ALL_EDGES;
  } 

  /*
   * For each edge, figure out the ID of the "other" vertex
   * and accumulate a union of 1 hop neighborhoods.
   */
  gather_type gather(icontext_type& context,
                     const vertex_type& vertex,
                     edge_type& edge) const {
    vector_push_gather gather;
    if(edge.data().sample_indicator == 0){
       gather.n.clear();
    }
    else{
      vid_vector othernbh = edge.target().id() == vertex.id() ?
                                       edge.source().data().vid_set : edge.target().data().vid_set;
      gather.n = othernbh;
    } 
   return gather;
  }


  /*
   * the gather result now contains the vertex IDs in the neighborhood.
   * store it on the vertex. 
   */
  void apply(icontext_type& context, vertex_type& vertex,
             const gather_type& neighborhoods) {
   do_not_scatter = false;
   if (neighborhoods.nbh_vec.size() == 0) {
     vertex.data().two_hop_set.clear();
     if (neighborhoods.n.size() != 0) {
       vertex.data().two_hop_set.push_back(neighborhoods.n);
     }
   }
   else {
     //insert at the end without merge
     vertex.data().two_hop_set.reserve(vertex.data().two_hop_set.size() + neighborhoods.nbh_vec.size());
     vertex.data().two_hop_set.insert(vertex.data().two_hop_set.end(),neighborhoods.nbh_vec.begin(), neighborhoods.nbh_vec.end());
   }
   
   
   //erase-remove this vertex id from all the 2hop neighborhoods
   std::vector<vid_vector>::iterator hopsit = vertex.data().two_hop_set.begin();
   while (hopsit != vertex.data().two_hop_set.end()) {
     if (hopsit->cset == NULL) {
       hopsit->vid_vec.erase( std::remove( hopsit->vid_vec.begin(), 
        hopsit->vid_vec.end(), vertex.id() ), hopsit->vid_vec.end() ); 
     }
     else {
       hopsit->cset->erase(vertex.id());
   }
   ++hopsit;
   }


   // vertex.data().vid_set.resize(std::distance(vertex.data().vid_set.begin(),newend));
   do_not_scatter = vertex.data().two_hop_set.empty();
  } // end of apply


  // No scatter
  edge_dir_type scatter_edges(icontext_type& context,
                             const vertex_type& vertex) const {
    return graphlab::NO_EDGES;
  }

};



vertex_data_type get_vertex_data(const graph_type::vertex_type& v) {
  return v.data();
}



  size_t get_edge_sample_indicator(const graph_type::edge_type& e){
           return e.data().sample_indicator;
   }

/*
 * A saver which saves a file where each line is a vid / neighborhood pair
 */
struct save_neighborhoods{
  std::string save_vertex(graph_type::vertex_type v) { 
  
  // char buf[1000];
  std::string buf = graphlab::tostr(v.id()) + "\n";
  if (v.data().vid_set.size() > 0) {
    buf += "1h: ";
    if (v.data().vid_set.cset == NULL) {
      for (size_t bt = 0; bt < v.data().vid_set.vid_vec.size(); bt++){
        buf += "\t" + graphlab::tostr(v.data().vid_set.vid_vec[bt]);
      }
    }
    else {
      for (graphlab::hopscotch_set<graphlab::vertex_id_type>::iterator bt=v.data().vid_set.cset->begin(); 
              bt != v.data().vid_set.cset->end(); bt++) {
        buf += "\t" + graphlab::tostr(*bt);
      }
    }
  }
  buf += "\n";

  if (v.data().two_hop_set.size() > 0) {
    buf += "2h: ";
    std::vector<vid_vector>::iterator hopit = v.data().two_hop_set.begin();
    while (hopit != v.data().two_hop_set.end()) {
      if (hopit->cset == NULL) {
        for (size_t bt = 0; bt < hopit->vid_vec.size(); bt++){
          buf += "\t" + graphlab::tostr(hopit->vid_vec[bt]);
        }
      }
      else {
        for (graphlab::hopscotch_set<graphlab::vertex_id_type>::iterator bt=hopit->cset->begin(); 
                bt != hopit->cset->end(); bt++) {
          buf += "\t" + graphlab::tostr(*bt);
        }
      }
      hopit++;
      buf += "\n";
    }
  }
  else {
  buf += "\n";
  
  }
  return buf;
  }
  std::string save_edge(graph_type::edge_type e) {
    return "";
  }
};


int main(int argc, char** argv) {

  graphlab::command_line_options clopts("Full 2-hop collection. "
    "For each vertex, return all vertices 1 hop and 2 hops away from it. "
    "The algorithm assumes that each undirected edge appears exactly once "
    "in the graph input. If edges may appear more than once, this procedure "
    "will over count.");
  std::string prefix, format;
  std::string per_vertex;
  clopts.attach_option("graph", prefix,
                       "Graph input. reads all graphs matching prefix*");
  clopts.attach_option("format", format,
                       "The graph format");
 clopts.attach_option("ht", HASH_THRESHOLD,
                       "Above this size, hash sets are used");
 // clopts.attach_option("ht2", HASH_THRESHOLD2,
 //                       "Above this size, unordered maps are used");
  clopts.attach_option("list_file", per_vertex,
                       "If not empty, will write the 1-hop and 2-hop "
                       "neighborhoods and "
                       "save to file with prefix \"[list_file]\". ");
 clopts.attach_option("sample_keep_prob", sample_prob_keep,
                        "Probability of keeping edge during sampling");
clopts.attach_option("sample_iter", sample_iter,
                       "Number of sampling iterations (global count)");
clopts.attach_option("min_prob", min_prob,
                       "min prob");
clopts.attach_option("max_prob", max_prob,
                       "max prob");
clopts.attach_option("prob_step", prob_step,
                       "prob step");

  if(!clopts.parse(argc, argv)) return EXIT_FAILURE;
  if (prefix == "") {
    std::cout << "--graph is not optional\n";
    clopts.print_description();
    return EXIT_FAILURE;
  }
  else if (format == "") {
    std::cout << "--format is not optional\n";
    clopts.print_description();
    return EXIT_FAILURE;
  }

  if ((per_vertex != "") & (sample_iter > 1)) {
    std::cout << "--multiple iterations only when no output list\n";
    clopts.print_description();
    return EXIT_FAILURE;
  }

  if (per_vertex != "") PER_VERTEX_COUNT = true;
  // Initialize control plane using mpi
  graphlab::mpi_tools::init(argc, argv);
  graphlab::distributed_control dc;

  //graphlab::launch_metric_server();
  // load graph
  graph_type graph(dc, clopts);
  graph.load_format(prefix, format);
  graph.finalize();
  dc.cout() << "Number of vertices: " << graph.num_vertices() << std::endl
            << "Number of edges (before sampling):    " << graph.num_edges() << std::endl;

  dc.cout() << "sample_prob_keep = " << sample_prob_keep << std::endl;
  dc.cout() << "sample_iter = " << sample_iter << std::endl;

  size_t reference_bytes;
  double total_time;
  double E2_time;
  // double E3_time;

  if(min_prob == max_prob){//i.e., they were not specified by user
    min_prob = sample_prob_keep;
    max_prob = sample_prob_keep;
  }

  double new_sample_prob = min_prob;
  while(new_sample_prob <= max_prob+0.00000000001){
    sample_prob_keep = new_sample_prob;
    new_sample_prob += prob_step;


  //START ITERATIONS HERE
  for (int sit = 0; sit < sample_iter; sit++) {

    reference_bytes = dc.network_bytes_sent();

    dc.cout() << "Iteration " << sit+1 << " of " << sample_iter << ". current sample prob: " << sample_prob_keep <<std::endl;

    graphlab::timer ti;
    
    // Initialize the vertex data
    graph.transform_vertices(init_vertex);

    //Sampling
    graph.transform_edges(sample_edge);
    //total_edges = graph.map_reduce_vertices<size_t>(get_vertex_degree)/2;
    total_edges = graph.map_reduce_edges<size_t>(get_edge_sample_indicator);
    dc.cout() << "Total edges counted (after sampling):" << total_edges << std::endl;


    // create engine to count the number of triangles
    dc.cout() << "Collecting hop 1..." << std::endl;
    graphlab::synchronous_engine<hop_count> engine1(dc, graph, clopts);
    // engine_type engine(dc, graph, clopts);


    engine1.signal_all();
    engine1.start();


    //dc.cout() << "Round 1 Counted in " << ti.current_time() << " seconds" << std::endl;
    
    //Sanity check for total edges count and degrees
    //total_edges = graph.map_reduce_vertices<size_t>(get_vertex_degree)/2;  
    //dc.cout() << "Total edges counted (after sampling) using degrees:" << total_edges << std::endl;

    //cannot put second engine before conditional?
    //graphlab::timer ti2;
    
    dc.cout() << "Collecting hop 2..." << std::endl;
    graphlab::timer ti2;
    graphlab::synchronous_engine<two_hop_count> engine2(dc, graph, clopts);
    engine2.signal_all();
    engine2.start();        
    E2_time = ti2.current_time();
    dc.cout() << "Full 2hop time: " << E2_time << "sec." << std::endl;
    // dc.cout() << "Clearing full 2hop ..." << std::endl;
    // graph.transform_vertices(clear_2hop_data);
    
    // dc.cout() << "Collecting 2nd moment..." << std::endl;
    // graphlab::timer ti3;
    // graphlab::synchronous_engine<moment_count> engine3(dc, graph, clopts);
    // engine3.signal_all();
    // engine3.start();
    // E3_time = ti3.current_time();
    // dc.cout() << "Second moment time: " << E3_time << "sec." << std::endl;
    // //distinct exact 2hop neighbors
    // double total_ct = graph.map_reduce_vertices<double>(get_tot_counts);
    // dc.cout() << "Total distinct 2hop neighbors: " << total_ct << std::endl;
    
      total_time = ti.current_time();
      dc.cout() << "Total runtime: " << total_time << "sec." << std::endl;

      std::ofstream myfile;
      char fname[30];
      sprintf(fname,"only_2_hop_times.txt");
      bool is_new_file = true;
      if (std::ifstream(fname)){
        is_new_file = false;
      }
      myfile.open (fname,std::fstream::in | std::fstream::out | std::fstream::app);
      // if(is_new_file) myfile << "#graph\tsample_prob_keep\t2hopTime\truntime\ttotal2counts" << std::endl;
      if(is_new_file) myfile << "#graph\tsample_prob_keep\t2hopTime\truntime" << std::endl;
      myfile << prefix << "\t"
             << sample_prob_keep << "\t"
             << std::setprecision (6)
             << E2_time << "\t"
             // << E3_time << "\t"
             << total_time << "\t"
             // << total_ct << "\t"
             << std::endl;

      myfile.close();

      sprintf(fname,"netw_only_2_hop_%d.txt",dc.procid());
      myfile.open (fname,std::fstream::in | std::fstream::out | std::fstream::app);

      myfile << dc.network_bytes_sent() - reference_bytes <<"\n";

      myfile.close();


    // }
    if (PER_VERTEX_COUNT==true) {
      graph.save(per_vertex,
              save_neighborhoods(),
              false, /* no compression */
              true, /* save vertex */
              false, /* do not save edge */
              1); /* one file per machine */
              // clopts.get_ncpus());

    }
    
    //dc.cout() << "Total Runtime: " << ti.current_time() << " sec" << std::endl;  

  }//for iterations
  }//while min/max_prob


  //graphlab::stop_metric_server();

  graphlab::mpi_tools::finalize();

  return EXIT_SUCCESS;
} // End of main

