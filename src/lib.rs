// http://semantic-domain.blogspot.co.uk/2013/11/antimirov-derivatives-for-regular.html

#[macro_use]
extern crate maplit;
extern crate bit_set;
extern crate dot;
use std::collections::{BTreeMap,BTreeSet, VecDeque, HashMap, HashSet};
use std::iter;
use std::mem;
use bit_set::BitSet;
use std::rc::Rc;
use std::ops::BitOr;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Re {
    Byte(char),
    Nil,
    Bot,
    Seq (RcRe, RcRe),
    Alt (RcRe, RcRe),
    Star (RcRe),
    Group (String, RcRe),
}

/*
let rec null = function
  | C _ | Bot -> false
  | Nil | Star _ -> true
  | Alt(r1, r2) -> null r1 || null r2
  | Seq(r1, r2) -> null r1 && null r2

   */

/*
    module R = Set.Make(struct type t = re let compare = compare end)
    let rmap f rs = R.fold (fun r -> R.add (f r)) rs R.empty

    module M = Map.Make(R)
    module I = Set.Make(struct type t = int let compare = compare end)
*/

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RcRe(Rc<Re>);


pub trait SemBuilder {
    /// Pushes an item onto the stack
    fn shift() -> Self;
    /// Notes that the top stack item has a name.
    fn grouped(self, name: String) -> Self;
    /// pops two items from the stack, and pushes their product.
    fn binary_product(self) -> Self;
}

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum Action {
    Noop,
    Push,
    Group(String),
    Product,
}

pub type ActionSeq = Rc<Vec<Action>>;

fn update_actions<T: Clone, F: FnOnce(&mut T)> (mut actions: Rc<T>, f: F) -> Rc<T> {
    f(Rc::make_mut(&mut actions));
    actions
}

impl SemBuilder for ActionSeq {
    fn shift () -> ActionSeq {
        Rc::new(vec![Action::Push])
    }
    fn grouped(self, name: String) -> ActionSeq {
        update_actions(self, move |r| r.push(Action::Group(name)))
    }
    fn binary_product(self) -> Self {
        update_actions(self, |r| r.push(Action::Product))
    }
}

impl SemBuilder for () {
    fn shift () -> () {
        ()
    }
    fn grouped(self, name: String) -> () {
        ()
    }
    fn binary_product(self) -> Self {
        ()
    }
}


pub trait Semantics<T> {
    fn apply(&self, c: char, ops: &T) -> Self;
}

impl<T> Semantics<T> for () {
    fn apply(&self, c: char, ops: &T) -> Self {
        ()
    }
}

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub struct ModelState(VecDeque<(Action, char)>);

use std::ops::Deref;
impl Semantics<ActionSeq> for ModelState {
    fn apply(&self, c: char, ops: &ActionSeq) -> Self {
        println!("Apply: {:?} to {:?}", ops, self);
        let mut m = self.clone();
        m.0.extend((*ops).iter().cloned().map(|o| (o, c)));
        m
    }
}

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub struct TransitionSet<A: Ord>(BTreeSet<(char, A, RcRe)>);

impl<A: SemBuilder + Clone + Ord> TransitionSet<A>{
    fn none() -> TransitionSet<A>{
        TransitionSet(btreeset!{})
    }
    fn unit(c: char, a: A, r: RcRe) -> TransitionSet<A>{
        TransitionSet(btreeset!{(c, a, r)})
    }
    fn union(&self, other: TransitionSet<A>) -> TransitionSet<A>{
        TransitionSet(&self.0 | &other.0)
    }

    fn prod(&self, t: RcRe) -> TransitionSet<A> {
        let res = match &*t.0 {
            &Re::Nil => btreeset!{},
            &Re::Bot => self.0.clone(),
            _ => self.0.iter()
                    .map(|&(x, ref a, ref p)| {
                            let derv = if p == &RcRe::nil() { t.clone() } else { RcRe::seq(p.clone(), t.clone()) };
                            (x, a.clone().binary_product(), derv)
                            }).collect()
        };
//         println!("prod: {:?} {:?} -> {:#?}", l, t, res);
        TransitionSet(res)
    }

    fn map_actions<F: Fn(A) -> A>(self, f: F) -> TransitionSet<A>{
        let ret = self.0.into_iter()
                        .map(|(c, a, re)| (c, f(a), re))
                        .collect();
        TransitionSet(ret)
    }
}

impl RcRe {
    fn is_null(&self) -> bool {
        match &*self.0 {
            &Re::Byte(_) | &Re::Bot => false,
            &Re::Nil | &Re::Star(_) => true,
            &Re::Alt(ref l, ref r) => l.is_null() || r.is_null(),
            &Re::Seq(ref l, ref r) => l.is_null() && r.is_null(),
            &Re::Group(_, ref r) => r.is_null(),
        }
    }
    pub fn nil() -> RcRe {
        RcRe(Rc::new(Re::Nil))
    }
    pub fn lit(c: char) -> RcRe {
        RcRe(Rc::new(Re::Byte(c)))
    }
    pub fn seq(l: RcRe, r: RcRe) -> RcRe {
        RcRe(Rc::new(Re::Seq(l, r)))
    }
    pub fn alt(l: RcRe, r: RcRe) -> RcRe {
        RcRe(Rc::new(Re::Alt(l, r)))
    }
    pub fn star(r: RcRe) -> RcRe {
        RcRe(Rc::new(Re::Star(r)))
    }
    pub fn group(self: RcRe, name: &'static str) -> RcRe {
        RcRe(Rc::new(Re::Group(name.to_string(), self)))
    }
    pub fn bottom() -> RcRe {
        RcRe(Rc::new(Re::Bot))
    }

    /// From Antimirov:
    ///
    ///
    pub fn lf<A: SemBuilder + Ord + Clone>(&self) -> TransitionSet<A> {
//         println!("lf: {:x} <- {:?}; null? {:?}", hash(&self), self, self.is_null());
        let res = match &*self.0 {
            &Re::Nil | &Re::Bot => TransitionSet::none(),
            &Re::Byte(m) => TransitionSet::unit(m, A::shift(), RcRe::nil()),
            &Re::Alt(ref l, ref r) => l.lf().union(r.lf()),
            &Re::Star(ref r) => r.lf().prod(self.clone()),
            &Re::Seq(ref l, ref r) => l.lf().prod(r.clone())
                .union(if !l.is_null() { TransitionSet::none() } else { r.lf() }),
            // TODO: Grouping
            &Re::Group(ref name, ref r) => r.lf().map_actions(|a|SemBuilder::grouped(a, name.clone())) ,
        };
//         println!("lf: {:x} {:?} -> {:#?}", hash(&self), self, res);
        res
    }
}

fn hash<T : ::std::hash::Hash>(x: T) -> u64 {
    use std::hash::*;
    let mut h = SipHasher::new();
    x.hash(&mut h);
    h.finish()
}

impl fmt::Debug for RcRe {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}", *self.0)
    }
}

impl std::ops::Mul for RcRe {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        RcRe::seq(self, other)
    }
}

impl std::ops::Add for RcRe {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        RcRe::alt(self, other)
    }
}


#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct NFA<A> {
    initial: usize,
    transition: BTreeMap<(usize, char), BTreeSet<(usize, A)>>,
    finals: BTreeSet<usize>,
    num_states: usize,
}

impl<A: fmt::Debug + Ord + SemBuilder + Clone> NFA<A> {
    /// From Antimirov:
    ///
    /// (PD[0], Δ[0], τ[0]) = (∅, {t}, ∅)
    ///
    /// PD[i+1] := PD[i] ∪  Δ[i]
    ///
    /// Δ[i+1] := ⋃(p∈ Δ[i]) {q | (x, q) ∈ lf(p)∧q ∉ PD[i+1]}
    /// τ[i+1] := τ[i]∪  {(p, x, q) | p ∈ Δ[i] ∧ (x, q) ∊ lf(p)}
    ///
    pub fn build(re: &RcRe) -> NFA<A> {
        #[derive(Debug, PartialEq, Eq)]
        struct State<A: Ord> {
            pd: BTreeSet<RcRe>,
            delta: BTreeSet<RcRe>,
            tau: BTreeSet<(RcRe, char, RcRe, A)>,
        }
        impl<A: Ord> State<A> {
            fn new() -> State<A> {
                State { pd: btreeset!{}, delta: btreeset!{}, tau: btreeset!{} }
            }
        }
        let mut state : State<A> = State::new();
        state.delta.insert(re.clone());

        loop {
            let mut next : State<A> = State::new();
//            println!("S: {:#?}", state);
            next.pd.extend(state.pd.iter().cloned());
            next.pd.extend(state.delta.iter().cloned());

            next.delta = state.delta.iter()
                .flat_map(|p| p.lf().0.into_iter().map(|(x, _, q): (_, A, _)| q))
                .filter(|q| !next.pd.contains(q))
                .collect();

            next.tau = state.tau.iter().cloned().chain(
                    state.delta.iter()
                    .flat_map(|p| p.lf().0.into_iter().map(move |(x, a, q)| (p.clone(), x, q, a))))
                    .collect();

//             println!("N@{:16x}: pd: {:?}; delta: {:?}; tau: {:?}",
//                  hash(&next), next.pd.len(), next.delta.len(), next.tau.len());

            if state == next { break; }
            state = next;
        }

        // let idx = state.pd.iter().enumerate().map(|(i, r)| (r.clone(), i)).collect::<BTreeMap<_, _>>();

        // let idx = state.pd.iter().enumerate().map(|(i, r)| (r.clone(), i)).collect::<BTreeMap<_, _>>();
        let initial = 0;
        let mut transitions = btreemap!{};
//         println!("digraph g{{");
//         println!("start -> N{};", initial);

        let mut idx = btreemap!{re.clone() => initial};
        let num_states = state.pd.len();

        let mut counter = 0;

        for (p, x, q, a) in state.tau {
//             println!("N{} -> N{} [label=\"{}\"];", idx[&p], idx[&q], x);
            let pi = *idx.entry(p.clone()).or_insert_with(|| { counter += 1; counter });
            let qi = *idx.entry(q.clone()).or_insert_with(|| { counter += 1; counter });

            let ent = transitions.entry((pi, x)).or_insert_with(|| btreeset!{});
            ent.insert((qi, a));
        }

        let finals = state.pd.iter().filter(|p| p.is_null()).map(|p| idx[p]).collect();

        NFA { transition: transitions, initial: initial, finals: finals, num_states: num_states }
    }

    pub fn matches(&self, s: &str) -> bool {
        self.parses::<()>(s).is_some()
    }

    pub fn parses<R: Semantics<A> + Ord + fmt::Debug + Clone + Default>(&self, s: &str) -> Option<R> {
        println!("Matching: {:?} against {:?}", s, self);
        let mut states = BTreeSet::new();
        let mut next = BTreeSet::new();
        states.insert((self.initial, Default::default()));
        for c in s.chars() {
            next.clear();
            println!("{:?} @ {:?}", states, c);
            for &(state, ref actions) in states.iter() {
                if let Some(ts) = self.transition.get(&(state, c)) {
                    next.extend(
                            ts.iter()
                            .map(|&(ns, ref a2)| {
                                (ns, Semantics::apply(actions, c, a2))
                            }));
                }
            }
            println!(" -> {:?}", next);
            std::mem::swap(&mut states, &mut next);
        }

        states.into_iter()
            .filter_map(|(s, a)| if self.finals.contains(&s) { Some(a) } else { None })
            .next()
    }
}

impl<'a, A: fmt::Debug> dot::Labeller<'a, usize, (usize, char, usize, A)> for NFA<A> {
     fn graph_id(&'a self) -> dot::Id<'a> { dot::Id::new("NFA").unwrap() }

     fn node_id(&'a self, n: &usize) -> dot::Id<'a> {
         dot::Id::new(format!("N{}", *n)).unwrap()
     }
     fn edge_label<'b>(&'b self, ed: &(usize, char, usize, A)) -> dot::LabelText<'b> {
         let &(_, c, _, ref a) = ed;
         dot::LabelText::LabelStr(format!("{:?} >- {:?}", c, a).into())
     }
}

use std::borrow::Cow;
impl<'a, A: Clone> dot::GraphWalk<'a, usize, (usize, char, usize, A)> for NFA<A> {
     fn nodes(&self) -> dot::Nodes<'a,usize> {
         let nodes = self.transition.keys()
             .map(|&(n, _c)| n).chain(self.finals.iter().cloned())
             .collect::<BTreeSet<_>>();

        let nodes = nodes.into_iter().collect::<Vec<usize>>();
        Cow::Owned(nodes)
     }

    fn edges(&'a self) -> dot::Edges<'a,(usize, char, usize, A)> {
        let edges = self.transition.iter()
            .flat_map(|(&(p, c), bs)| bs.iter().map(move |&(q, ref a)| (p, c, q, a.clone())))
            .collect();
        Cow::Owned(edges)
    }
    fn source(&self, e: &(usize, char, usize, A)) -> usize { let &(s,_,_,_) = e; s }

    fn target(&self, e: &(usize, char, usize, A)) -> usize { let &(_,_,t,_) = e; t }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
struct DFA {
    size: usize,
    fail: usize,
    trans: Vec<(usize, char, usize)>,
    finalp: BTreeSet<usize>,
}

/*
fn charfold<F: Fn(V, char) -> V, V>(f: F, zero: V) -> V {
    (0char..256).fold(zero, f)
}

impl DFA{
    fn build(re: RcRe) -> DFA {
        let mut dfa : DFA = Default::default();
        let mut states = BTreeMap::new();
        let mut queue = VecDeque::new();

        queue.push_back(re);

        while let Some(exp) = queue.pop_front() {
            // let exp = exp.normalize();
            // LookupOrInsert
            let idx = states.len();
            let idx = *states.entry(exp.clone()).or_insert(idx);
            if exp.is_null() {
                dfa.finalp.insert(idx);
            }
        }

        dfa
    }
}
*/


#[cfg(test)]
mod tests {
    use super::{RcRe,NFA, Action, ActionSeq, ModelState};
    use std::rc::Rc;
    #[test]
    fn should_build_nfa() {
        use super::RcRe as R;
        let re = R::lit('a') * R::lit('b') * R::lit('a') * R::lit('b') +
                R::lit('a') * R::lit('b') * R::lit('b') * R::lit('b');
//         println!("Re: {:?}", re);
        let nfa = NFA::<()>::build(&re);
//         println!("NFA: {:?}", nfa);
//         println!("---");
        assert!(nfa.matches("abbb"));
//         println!("---");
        assert!(nfa.matches("abab"));
//         println!("---");
        assert!(!nfa.matches("abaa"));
        assert!(!nfa.matches("aba"));
    }

    #[test]
    fn should_build_nfa2() {
        use super::RcRe as R;
        let re = R::star(R::lit('a') * R::lit('b')) * R::lit('c');
//         println!("Re: {:?}", re);
        let nfa = NFA::<()>::build(&re);
         println!("NFA: {:?}", nfa);
         println!("--- abc");
        assert!(nfa.matches("abc"));
         println!("--! aabbc");
        assert!(!nfa.matches("aabbc"));
//         println!("---");
        assert!(nfa.matches("c"));
//         println!("---");
        assert!(nfa.matches("ababc"));
        assert!(!nfa.matches("aba"));
    }
    #[test]
    fn should_build_nfa3() {
        use super::RcRe as R;
        let re = R::lit('a') * R::lit('b') * R::lit('c') + R::lit('e') * R::lit('d') * R::lit('c') ;
//         println!("Re: {:?}", re);
        let nfa = NFA::<()>::build(&re);
//         println!("NFA: {:?}", nfa);
//         println!("---");
        assert!(nfa.matches("abc"));
        assert!(nfa.matches("edc"));
        assert!(!nfa.matches("bc"));
        assert!(!nfa.matches("dc"));
        assert!(!nfa.matches("c"));
    }
    #[test]
    fn should_build_nfa4() {
        use super::RcRe as R;
        let re = R::lit('a') * R::lit('e');
//         println!("Re: {:?}", re);
        let nfa = NFA::<ActionSeq>::build(&re);
//         println!("NFA: {:?}", nfa);
//         println!("---");
        assert!(nfa.matches("ae"));
        assert!(!nfa.matches("e"));
        assert!(!nfa.matches("ea"));

        // assert!(false);
    }

    #[test]
    fn should_build_with_groups() {
        use super::RcRe as R;
        let re = (R::lit('a') + R::lit('a') * R::lit('b')).group("left") *
            (R::lit('d') + R::lit('d') * R::lit('e')).group("right");
        println!("Re: {:?}", re);
        let nfa = NFA::<()>::build(&re);
        println!("NFA: {:?}", nfa);
        println!("--- ad");
        let ad = nfa.parses::<()>("ad");
        println!("=> {:?}", ad);
        assert!(ad.is_some());

        println!("--- abde");
        let abde = nfa.parses::<()>("abde");
        println!("=> {:?}", abde);
        assert!(abde.is_some());

        println!("--- abd");
        assert!(nfa.matches("abd"));
        println!("--- abe");
        assert!(!nfa.matches("abe"));
        // assert!(false);
    }

    fn pathological(n: usize) -> (RcRe, String) {
        use super::RcRe as R;
        use std::iter;
        // a?^na^n matches a^n:
        let a = R::lit('a');
        let a_maybe = a.clone() + R::nil();

        let re =
            iter::repeat(a_maybe).take(n).fold(R::nil(), |a, b| a * b) *
            iter::repeat(a).take(n).fold(R::nil(), |a, b| a * b);

        (re, iter::repeat('a').take(n).collect())
    }

    #[test]
    fn should_match_pathological_example() {
        let (re, s) = pathological(8);
        println!("RE: {:#?}", re);
        let nfa = NFA::<()>::build(&re);
        println!("NFA: {:#?}", nfa);
        assert!(nfa.matches(&s));
        // assert!(false);
    }

    #[test]
    fn nfa_size() {
        use dot;
        use std::fs::File;
        for n in 0..8 {
            println!("a?^{0}a^{0} matches a^{0}", n);
            let mut f = File::create(&format!("target/pathological-{:04}.dot", n)).expect("open dot file");
            let (re, s) = pathological(n);
            let nfa = NFA::<()>::build(&re);
            dot::render(&nfa, &mut f).expect("render dot");
        }
    }

    #[test]
    fn should_parse_arithmetic() {
        use dot;
        use std::fs::File;
        use super::RcRe as R;
        let digits = (0..10).map(|d| (d + '0' as u8) as char).map(R::lit).fold(R::bottom(), |acc, x| acc + x);
        let number = (digits.clone() * R::star(digits)).group("number");
        let products = (number.clone() * R::star(R::lit('*') * number.clone())).group("products");
        let expr = (products.clone() * R::star(R::lit('+') * products.clone())).group("sums");
        println!("Re: {:?}", expr);
        let nfa = NFA::<ActionSeq>::build(&expr);
        println!("NFA: {:?}", nfa);
        let mut f = File::create(&"target/arithmetic.dot").expect("open dot file");
        dot::render(&nfa, &mut f).expect("render dot");

        println!("--- 0");
        let zero = nfa.parses::<ModelState>("0");
        println!("=> {:?}", zero);
        assert!(zero.is_some());

        println!("--- 12");
        let twelve = nfa.parses::<ModelState>("12");
        println!("=> {:?}", twelve);
        assert!(twelve.is_some());

        println!("--- 12*2+3");
        let expr = nfa.parses::<ModelState>("12*2+3");
        println!("=> {:?}", expr);
        assert!(expr.is_some());


        // assert!(false);
    }
 
}
