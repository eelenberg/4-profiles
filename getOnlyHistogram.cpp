/*
getOnlyHistogram.cpp Compute 2-hop histogram
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
//#include <boost/multiprecision/cpp_int.hpp>
#include <graphlab.hpp>
#include <graphlab/ui/metrics_server.hpp>
#include <graphlab/util/hopscotch_set.hpp>
#include <graphlab/macros_def.hpp>

//using namespace boost::multiprecision;

/**
 *
 *  
 * In this program we implement the "hash-set" version of the
 * "edge-iterator" algorithm described in
 * 
 *    T. Schank. Algorithmic Aspects of Triangle-Based Network Analysis.
 *    Phd in computer science, University Karlsruhe, 2007.
 *
 * The procedure is quite straightforward:
 *   - each vertex maintains a list of all of its neighbors in a hash set.
 *   - For each edge (u,v) in the graph, count the number of intersections
 *     of the neighbor set on u and the neighbor set on v.
 *   - We store the size of the intersection on the edge.
 * 
 * This will count every triangle exactly 3 times. Summing across all the
 * edges and dividing by 3 gives the desired result.
 *
 * The preprocessing stage take O(|E|) time, and it has been shown that this
 * algorithm takes $O(|E|^(3/2))$ time.
 *
 * If we only require total counts, we can introduce a optimization that is
 * similar to the "forward" algorithm
 * described in thesis above. Instead of maintaining a complete list of all
 * neighbors, each vertex only maintains a list of all neighbors with
 * ID greater than itself. This implicitly generates a topological sort
 * of the graph.
 *
 * Then you can see that each triangle
 *
 * \verbatim
  
     A----->C
     |     ^
     |   /
     v /
     B
   
 * \endverbatim
 * Must be counted only once. (Only when processing edge AB, can one
 * observe that A and B have intersecting out-neighbor sets).
 */
 

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
size_t HASH_THRESHOLD2 = 45000; //for nbh_count

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

struct idcount {
 graphlab::vertex_id_type vert_id;
 size_t count;
 //will this prevent overflow?
 // double count;
// Code to serialize and deserialize the structure. But not sure if this right ! 
 void save(graphlab::oarchive &oarc) const {
    oarc << vert_id << count; }
  void load(graphlab::iarchive &iarc) {
    iarc >> vert_id>>count; }
};
struct twoids{
   graphlab::vertex_id_type first;
   graphlab::vertex_id_type second;
 void save(graphlab::oarchive &oarc) const {
    oarc << first << second; }
  void load(graphlab::iarchive &iarc) {
    iarc >> first>>second; }
};

// The compare_idcount funcion helps sort the structure.
bool compare_idcount (const idcount& a, const idcount& b){  
 return (a.vert_id < b.vert_id);
}
bool operator==(const idcount& i, const graphlab::vertex_id_type val) {
  return (i.vert_id == val);
}

//NEW: MODEL nbh_count AFTER vid_vector, but the basic unit is a <id,count> idcount
struct nbh_count{
  std::vector<idcount> idc_vec;
  // boost::unordered_map<graphlab::vertex_id_type,size_t> *idc_map;
  boost::unordered_map<graphlab::vertex_id_type,size_t> idc_map; //map instead of pointer?
 
  nbh_count& operator=(const nbh_count& n) {
    if (this == &n) return *this;
    idc_vec = n.idc_vec;
    // if (n.idc_map != NULL) {
    if (!(n.idc_map.empty())) {
      // allocate the unordered map if the other side is using an unordered map
      // or clear if I alrady have one
      if (idc_map.empty()) {
        //does this work, or reserve size in another line?
        // idc_map = new boost::unordered_map<graphlab::vertex_id_type,size_t>(HASH_THRESHOLD2);
      }
      else {
        // idc_map->clear();
        idc_map.clear();
      }
      //idc_map.reserve(n.idc_map.size()); //not necessary before a copy assign?
      idc_map = n.idc_map;
    }
    else {
      // if the other side is not using an unordered map, lets not use an unordered map
      // either
      if (!(idc_map.empty())) {
        // delete idc_map;
        // idc_map = NULL;
        idc_map.clear();
      }
    }
    return *this;
  }

  ~nbh_count() {
    // if (idc_map != NULL) delete idc_map;
    idc_vec.clear();
    if (!(idc_map.empty())) idc_map.clear();
 //     std::cout<<" Destroyed "<<std::endl;
  }

  // assigns a vector of idcounts to this storage.
  // this function will clear the contents of the vid_vector
  // and reconstruct it.
  // If the assigned values has length >= HASH_THRESHOLD,
  // we will allocate a cuckoo set to store it. Otherwise,
  // we just store a sorted vector
  //this should be used in the nbc_intersection_complement
  void assign(const std::vector<idcount>& vec) {
    clear();
    if (vec.size() >= HASH_THRESHOLD2) {
        // move to cset
        // idc_map = new boost::unordered_map<graphlab::vertex_id_type,size_t>(HASH_THRESHOLD2);
        // idc_map.reserve(HASH_THRESHOLD2);
        idc_map.reserve(vec.size());
        // foreach (idcount b, vec) {
        //   idc_map[b.vert_id] += b.count;
        // }
        std::vector<idcount>::const_iterator vit=vec.begin();
        while(vit != vec.end()) {
          idc_map[vit->vert_id] += vit->count;
          ++vit;
        }
    }
    else {
      idc_vec = vec;
      idc_map.clear();
       std::sort(idc_vec.begin(), idc_vec.end(), compare_idcount);
    }
  }

  //make this a += operator instead??
  void merge_many(const std::vector<idcount>& ovec) {//idcount, iterator to vector<idcount> vit (output vector or map)
    size_t num= idc_vec.size()+ovec.size();
    if (idc_map.empty() && (num  < HASH_THRESHOLD2)) {

        std::vector<idcount> a;
        a.reserve(num);


      std::merge(idc_vec.begin(),idc_vec.end(),ovec.begin(),ovec.end(),std::back_inserter(a),compare_idcount); 
      
      idc_vec.clear();
      idc_vec.reserve(num);

      std::vector<idcount>::iterator it; 
      std::vector<idcount>::iterator it3;  
      for (it=a.begin(); it!=a.end();it++) {
        it3=it;
        it3++;
        if((it3!=a.end()) &&  (it3->vert_id == it->vert_id)) {
          it->count=it->count+it3->count;
          idc_vec.push_back(*it++);
        }                   
        else{
          idc_vec.push_back(*it);
        }
      }
      //a.clear();
    }
    else {
      //output is unordered_map
      if (idc_map.empty()) {
        //convert from vector to unordered_map
        idc_map.reserve(num);
        std::vector<idcount>::iterator it = idc_vec.begin();
        while (it!=idc_vec.end()) {
       // KARTHIK CHANGE - RHS of the following assignment was 1 for some reason.
          idc_map[it->vert_id] = it->count;
          ++it;
        }
        idc_vec.clear();
      }
      //add vector to the unordered_map
      std::vector<idcount>::const_iterator it = ovec.begin();
      while (it!=ovec.end()) {
        idc_map[it->vert_id] += it->count;
        ++it;
      }
    }
  }
  
  void merge_many(const boost::unordered_map<graphlab::vertex_id_type,size_t>& mpt) {//idcount, iterator to unordered_map mit (output map)
    //if currently vector, copy over map and add vector to it
    if (idc_map.empty()) {
      // boost::unordered_map<graphlab::vertex_id_type,size_t> const_iterator mit;
      idc_map.insert(mpt.begin(), mpt.end());
      std::vector<idcount>::iterator it; 
      for (it = idc_vec.begin(); it!= idc_vec.end(); it++) {
        idc_map[it->vert_id] += it->count;
      }
      idc_vec.clear();
    }
    //if currently map, iterate over input and insert
    else {
      boost::unordered_map<graphlab::vertex_id_type,size_t>::const_iterator mit = mpt.begin();
      while (mit!=mpt.end()){
        idc_map[mit->first] += mit->second;
        ++mit;
      }
    }
  }
  
  //just load both, whats the big deal of not a pointer?
  void save(graphlab::oarchive& oarc) const {
    oarc << idc_vec;
    oarc << idc_map;
  }
  void clear() {
    idc_vec.clear();
    // if (!(idc_map.empty())) {
      idc_map.clear();
    // }
  }
  size_t size() const {
    return idc_map.empty() ? idc_vec.size() : idc_map.size();
  }
  void load(graphlab::iarchive& iarc) {
    iarc >> idc_vec;
    iarc >> idc_map;
  }
};



// TWO-HOP OPTIMIZATION CHANGE
// A NEW FUNCTION THAT COMPUTES THE RESULT OF THE SMALL_SET (INTERSECTION) LARGE_SET's COMPLEMENT and that does not contain one other vertex id  and returns a vector of ID counts.
//use assign instead of append 1, this is the first time the gather is assigned
void nbh_intersect_complement(const vid_vector& smaller_set, const vid_vector& larger_set,graphlab::vertex_id_type p, nbh_count& difflist) {
    std::vector<graphlab::vertex_id_type> temp;
    std::vector<idcount> diffvec;
    diffvec.clear();
    difflist.clear();
    temp.clear();
    if (smaller_set.cset == NULL && larger_set.cset == NULL) {
        std::set_difference(smaller_set.vid_vec.begin(), smaller_set.vid_vec.end(),
                            larger_set.vid_vec.begin(), larger_set.vid_vec.end(),
                            std::back_inserter(temp));
        // difflist.assign(temp); //must be an idcount, not just ids?
        diffvec.reserve(temp.size());
        for (size_t k=0;k<temp.size();k++) {
            if(p!=temp.at(k)) {
            idcount a;
            a.vert_id=temp.at(k);
            a.count=1;
            diffvec.push_back(a);
            }
        }
       temp.clear(); // KARTHIK CHANGE- clearing temp vector. 
    }
    else if (smaller_set.cset == NULL && larger_set.cset != NULL) {
        
        diffvec.reserve(smaller_set.vid_vec.size());
        foreach(graphlab::vertex_id_type vid, smaller_set.vid_vec) {
            if( larger_set.cset->count(vid)==0 && p!=vid) {
                idcount a;
                a.vert_id=vid;
                a.count=1;
                diffvec.push_back(a);
            }
        }
    }
    else if (smaller_set.cset != NULL && larger_set.cset == NULL) {
        
        diffvec.reserve(larger_set.vid_vec.size());
        foreach(graphlab::vertex_id_type vid, *(smaller_set.cset)) {
            if( count(larger_set.vid_vec.begin(),larger_set.vid_vec.end(),vid)==0 && p!=vid) {
                idcount a;
                a.vert_id=vid;
                a.count=1;
                diffvec.push_back(a);                
            }
        }        
    }
    else {
        diffvec.reserve(smaller_set.cset->size());
        foreach(graphlab::vertex_id_type vid, *(smaller_set.cset)) {
            if( larger_set.cset->count(vid)==0 && p!=vid) {
                idcount a;
                a.vert_id=vid;
                a.count=1;
                diffvec.push_back(a);
            }
        }        
    }
    difflist.assign(diffvec);   
    diffvec.clear(); // KARTHIK CHANGE- diffvec vector is cleared
}


/*
 * Each vertex maintains a list of all its neighbors.
 * and a final count for the number of triangles it is involved in
 */
struct vertex_data_type {
  //vertex_data_type(): num_triangles(0){ }
  // A list of all its neighbors
  vid_vector vid_set;
  // std::vector<vid_vector> two_hop_set; //things that are at most 2 hops away, can always remove 1 hop afterwards?
  double num_moments;
  // double sum_counts;
  
  void save(graphlab::oarchive &oarc) const {
    oarc << vid_set << num_moments;
  }
  void load(graphlab::iarchive &iarc) {
    iarc >> vid_set >> num_moments;
  }
};



/*
 * Each edge is simply a counter of triangles
 *
 */
//typedef uint32_t edge_data_type;

//NEW EDGE DATA AND GATHER
struct edge_data_type {
  bool sample_indicator;
  void save(graphlab::oarchive &oarc) const {
    oarc << sample_indicator;
    //oarc << vid_set << num_triangles;
    // oarc << n1 << n2 << n2e << n2c << n3 << sample_indicator;
  }
  void load(graphlab::iarchive &iarc) {
    iarc >> sample_indicator;
    // iarc >> n1 >> n2 >> n2e >> n2c >> n3 >> sample_indicator;
  }
};



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
    // std::cout << "CURRENT SIZE: " << vid_vec.size() << " INCOMING SIZE: " << other.vid_vec.size() << "\n"; //WHY IS THIS 0??
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
    //does this check for duplicates??
    if (other.vid_vec.size() > 0) {
      size_t ct = vid_vec.size();
      vid_vec.resize(vid_vec.size() + other.vid_vec.size());
      // std::cout << "OLD SIZE: " << ct << " NEW SIZE: " << vid_vec.size() << "\n";
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




struct moment_gather {
  nbh_count b;

  moment_gather& operator+=(const moment_gather& other) {
    
    //try with no singletons, just vectors and umaps
    //how did the cset get appended? now we need to merge instead
    if (b.size() == 0) {
      b = other.b; 
      return (*this);
    }
    else if (other.b.size() == 0) {
      return *this;
    }

    if (!(other.b.idc_map.empty())) {
      b.merge_many(other.b.idc_map);
    }    
    else if (other.b.idc_vec.size() > 0) {
      b.merge_many(other.b.idc_vec);
    }
    
    return *this;
  }

  // serialize
  void save(graphlab::oarchive& oarc) const {
    oarc << b;
  }

  void load(graphlab::iarchive& iarc) {
    iarc >> b;
  }
  ~moment_gather() {
    //std::cout<<"Getting destroyed "<<std::endl;
   //  b.~nbh_count();
   }
};


//another sets_union_gather to union 2 sets instead of vertex and set???
//no vertex_id_type
//or just overload this operator?
//vectors_union_gather??
// graphlab::hopscotch_set<graphlab::vertex_id_type> *cset;


/*
 * Define the type of the graph
 */
typedef graphlab::distributed_graph<vertex_data_type,
                                    edge_data_type> graph_type;


// //move init outside constructor (must be declared after graph_type)
void init_vertex(graph_type::vertex_type& vertex) { 
     vertex.data().vid_set.clear();
     vertex.data().num_moments = 0;
     // vertex.data().sum_counts = 0;
}

// void clear_2hop_data(graph_type::vertex_type& vertex) { 
//      vertex.data().two_hop_set.clear();
// }

  void sample_edge(graph_type::edge_type& edge) {
     if(graphlab::random::rand01() < sample_prob_keep)   
       edge.data().sample_indicator = 1;
     else
       edge.data().sample_indicator = 0;
  }


/*
 * This class implements the triangle counting algorithm as described in
 * the header. On gather, we accumulate a set of all adjacent vertices.
 * If per_vertex output is not necessary, we can use the optimization
 * where each vertex only accumulates neighbors with greater vertex IDs.
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



class moment_count :
      public graphlab::ivertex_program<graph_type, moment_gather>,
      /* I have no data. Just force it to POD */
      public graphlab::IS_POD_TYPE  {
public:
  // Gather on all edges
  edge_dir_type gather_edges(icontext_type& context,
                             const vertex_type& vertex) const {
    return graphlab::ALL_EDGES;
  }
  // We gather the number of triangles each edge is involved in
  // size_t gather(icontext_type& context,
  gather_type gather(icontext_type& context,
                     const vertex_type& vertex,
                     edge_type& edge) const {
    moment_gather gather;


    if (edge.data().sample_indicator == 1){
      //return edge.data();
      

      // std::cout << "edge.data().n2e = "<<edge.data().n2e<<" edge.data().n2c = "<<edge.data().n2c<<"\n";
      if (vertex.id() == edge.source().id()){        
        const vertex_data_type& targetlist = edge.target().data();
        nbh_intersect_complement(targetlist.vid_set,edge.source().data().vid_set,edge.source().id(),gather.b);
      }

      else{
        const vertex_data_type& srclist = edge.source().data();
        nbh_intersect_complement(srclist.vid_set,edge.target().data().vid_set,edge.target().id(),gather.b);
      }
    }


    else{
      gather.b.clear();
    }
    return gather;
  }

  /* the gather result is the total sum of the number of triangles
   * each adjacent edge is involved in . Dividing by 2 gives the
   * desired result.
   */
  void apply(icontext_type& context, vertex_type& vertex,
             const gather_type& ecounts) {
    
 // Now using the gathered information from ecounts.b to form u[9].  So every id from there has to be checked if it is a member of the vertex's neighbor and only if not count choose 2 must be 
 //  aggregated.
     vertex.data().num_moments=0;
     // vertex.data().sum_counts=0;
     if (ecounts.b.idc_map.empty()) {
       // std::cout << "vertex id: " << vertex.id() << ", degree: " << vertex.data().vid_set.size() << ", size of idc_vec: " << ecounts.b.idc_vec.size() << std::endl;
       // std::vector<idcount> e1=ecounts.b.idc_vec;
       std::vector<idcount>::const_iterator it = ecounts.b.idc_vec.begin();
       while (it!=ecounts.b.idc_vec.end()){
         vertex.data().num_moments+= (it->count * (it->count - 1))/2; // doing count choose 2 if the id is not the vertex neighbor.         
         // vertex.data().sum_counts+= it->count - 1; 
         ++it;
       }
     }

     else {
      // std::cout << "vertex id: " << vertex.id() << ", degree: " << vertex.data().vid_set.size() << ", size of idc_map: " << ecounts.b.idc_map.size() << std::endl;
      boost::unordered_map<graphlab::vertex_id_type,size_t>::const_iterator it = ecounts.b.idc_map.begin();
      while (it!=ecounts.b.idc_map.end()){
         vertex.data().num_moments+= (it->second * (it->second - 1))/2; // doing count choose 2 if the id is not the vertex neighbor.
         ++it;
      }
    }     
  }

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
 * A saver which saves a file where each line is a vid / # triangles pair
 */
// struct save_neighborhoods{
//   std::string save_vertex(graph_type::vertex_type v) {   
//     // char buf[1000];
//     std::string buf = graphlab::tostr(v.id()) + "\n";
//     //print different things based on cset, ideally we should make an iterator for the class vid_vector
//     if (v.data().vid_set.size() > 0) {
//       buf += "1h: ";
//       if (v.data().vid_set.cset == NULL) {
//         for (size_t bt = 0; bt < v.data().vid_set.vid_vec.size(); bt++){
//           buf += "\t" + graphlab::tostr(v.data().vid_set.vid_vec[bt]);
//         }
//       }
//       else {
//         for (graphlab::hopscotch_set<graphlab::vertex_id_type>::iterator bt=v.data().vid_set.cset->begin(); 
//                 bt != v.data().vid_set.cset->end(); bt++) {
//           buf += "\t" + graphlab::tostr(*bt);
//         }
//       }
//     }
//     buf += "\n";

    
//     return buf;
//   }
//   std::string save_edge(graph_type::edge_type e) {
//     return "";
//   }
// };


int main(int argc, char** argv) {

  graphlab::command_line_options clopts("2-hop histogram calculation. "
    "For each vertex, compute the 2-hop histogram (limited 2-hop information). "
    "The algorithm assumes that each undirected edge appears exactly once "
    "in the graph input. If edges may appear more than once, this procedure "
    "will over count.");
  std::string prefix, format;
  clopts.attach_option("graph", prefix,
                       "Graph input. reads all graphs matching prefix*");
  clopts.attach_option("format", format,
                       "The graph format");
 clopts.attach_option("ht", HASH_THRESHOLD,
                       "Above this size, hash sets are used");
 clopts.attach_option("ht2", HASH_THRESHOLD2,
                       "Above this size, unordered maps are used");
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
  // double E2_time;
  double E3_time;

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
   
    dc.cout() << "Collecting 2nd moment..." << std::endl;
    graphlab::timer ti3;
    graphlab::synchronous_engine<moment_count> engine3(dc, graph, clopts);
    engine3.signal_all();
    engine3.start();
    E3_time = ti3.current_time();
    dc.cout() << "Second moment time: " << E3_time << "sec." << std::endl;
    //distinct exact 2hop neighbors
    // double total_ct = graph.map_reduce_vertices<double>(get_tot_counts);
    // dc.cout() << "Total distinct 2hop neighbors: " << total_ct << std::endl;
    
      total_time = ti.current_time();
      dc.cout() << "Total runtime: " << total_time << "sec." << std::endl;

      std::ofstream myfile;
      char fname[30];
      sprintf(fname,"only_histogram_times.txt");
      bool is_new_file = true;
      if (std::ifstream(fname)){
        is_new_file = false;
      }
      myfile.open (fname,std::fstream::in | std::fstream::out | std::fstream::app);
      // if(is_new_file) myfile << "#graph\tsample_prob_keep\tht2\tmomentTime\truntime\ttotal2counts" << std::endl;
      if(is_new_file) myfile << "#graph\tsample_prob_keep\th2\tmomentTime\truntime" << std::endl;
      myfile << prefix << "\t"
             // << (global_counts.num_triangles/3)/pow(sample_prob_keep, 3) << "\t"
             // << (global_counts.num_wedges/3)/pow(sample_prob_keep, 2) - (global_counts.num_triangles/3)*(1-sample_prob_keep)/pow(sample_prob_keep, 3) << "\t"
             // << (global_counts.num_disc/3)/sample_prob_keep - (global_counts.num_wedges/3)*(1-sample_prob_keep)/pow(sample_prob_keep, 2) << "\t"
             // << (global_counts.num_empty/3)-(global_counts.num_disc/3)*(1-sample_prob_keep)/sample_prob_keep  << "\t"
             << sample_prob_keep << "\t"
             << HASH_THRESHOLD2 << "\t"
             << std::setprecision (6)
             // << E2_time << "\t"
             << E3_time << "\t"
             << total_time << "\t"
             // << total_ct << "\t"
             << std::endl;

      myfile.close();

      sprintf(fname,"netw_only_histogram_%d.txt",dc.procid());
      myfile.open (fname,std::fstream::in | std::fstream::out | std::fstream::app);

      myfile << dc.network_bytes_sent() - reference_bytes <<"\n";

      myfile.close();


    
    //dc.cout() << "Total Runtime: " << ti.current_time() << " sec" << std::endl;  

  }//for iterations
  }//while min/max_prob


  //graphlab::stop_metric_server();

  graphlab::mpi_tools::finalize();

  return EXIT_SUCCESS;
} // End of main

