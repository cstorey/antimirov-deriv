// http://semantic-domain.blogspot.co.uk/2013/11/antimirov-derivatives-for-regular.html

#[macro_use]
extern crate maplit;
extern crate bit_set;
use std::collections::{BTreeMap,BTreeSet, VecDeque, HashMap, HashSet};
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

impl RcRe {
    fn is_null(&self) -> bool {
        match &*self.0 {
            &Re::Byte(_) | &Re::Bot => false,
            &Re::Nil | &Re::Star(_) => true,
            &Re::Alt(ref l, ref r) => l.is_null() || r.is_null(),
            &Re::Seq(ref l, ref r) => l.is_null() && r.is_null(),
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

    /// From Antimirov:
    ///
    ///
    pub fn lf(&self) -> BTreeSet<(char, RcRe)> {
//         println!("lf: {:x} <- {:?}; null? {:?}", hash(&self), self, self.is_null());
        let res = match &*self.0 {
            &Re::Nil | &Re::Bot => btreeset!{},
            &Re::Byte(m) => btreeset!{(m, RcRe::nil())},
            &Re::Alt(ref l, ref r) => &l.lf() | &r.lf(),
            &Re::Star(ref r) => Self::prod(r.lf(), self.clone()),
            &Re::Seq(ref l, ref r) => Self::prod(l.lf(), r.clone())
                .union(&if !l.is_null() { btreeset!{} } else { r.lf() })
                .cloned()
                .collect()
        };
//         println!("lf: {:x} {:?} -> {:#?}", hash(&self), self, res);
        res
    }

    fn prod(l: BTreeSet<(char, RcRe)>, t: RcRe) -> BTreeSet<(char, RcRe)> {
        let res = match &*t.0 {
            &Re::Nil => btreeset!{},
            &Re::Bot => l.clone(),
            _ => l.clone().into_iter()
                    .map(|(x, ref p)| (x, if p == &Self::nil() { t.clone() } else { Self::seq(p.clone(), t.clone()) })).collect()
        };
//         println!("prod: {:?} {:?} -> {:#?}", l, t, res);
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
pub struct NFA {
    initial: usize,
    transition: BTreeMap<(usize, char), BitSet>,
    finals: BTreeSet<usize>,
}

impl NFA {
    /// From Antimirov:
    ///
    /// (PD[0], Δ[0], τ[0]) = (∅, {t}, ∅)
    ///
    /// PD[i+1] := PD[i] ∪  Δ[i]
    ///
    /// Δ[i+1] := ⋃(p∈ Δ[i]) {q | (x, q) ∈ lf(p)∧q ∉ PD[i+1]}
    /// τ[i+1] := τ[i]∪  {(p, x, q) | p ∈ Δ[i] ∧ (x, q) ∊ lf(p)}
    ///
    pub fn build(re: &RcRe) -> NFA {
        #[derive(Debug, Default, PartialEq, Eq)]
        struct State {
            pd: HashSet<RcRe>,
            delta: HashSet<RcRe>,
            tau: HashSet<(RcRe, char, RcRe)>,
        }
        let mut state : State = Default::default();
        state.delta.insert(re.clone());

        loop {
            let mut next : State = Default::default();
//            println!("S: {:#?}", state);
            next.pd.extend(state.pd.iter().cloned());
            next.pd.extend(state.delta.iter().cloned());

            next.delta = state.delta.iter()
                .flat_map(|p| p.lf().into_iter().map(|(x, q)| q))
                .filter(|q| !next.pd.contains(q))
                .collect();

            next.tau = state.tau.iter().cloned().chain(
                    state.delta.iter()
                    .flat_map(|p| p.lf().into_iter().map(move |(x, q)| (p.clone(), x, q))))
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

        let mut counter = 0;

        for (p, x, q) in state.tau {
//             println!("N{} -> N{} [label=\"{}\"];", idx[&p], idx[&q], x);
            let pi = *idx.entry(p.clone()).or_insert_with(|| { counter += 1; counter });
            let qi = *idx.entry(q.clone()).or_insert_with(|| { counter += 1; counter });

            let ent = transitions.entry((pi, x)).or_insert_with(|| BitSet::new());
            ent.insert(qi);
        }
        for state in state.pd.iter() {
//             println!("N{}[label=\"{:?}\"];", idx[state], state);
        }
//         println!("}}");

        let finals = state.pd.iter().filter(|p| p.is_null()).map(|p| idx[p]).collect();

        NFA { transition: transitions, initial: initial, finals: finals }
    }

    pub fn matches(&self, s: &str) -> bool {
//         // println!("Matching: {:?} against {:?}", s, self);
        let mut states = BitSet::new(); states.insert(self.initial);
        let mut next = BitSet::new();
        for c in s.chars() {
            next.clear();
            // print!("{:?} @ {:?}", states, c);
            for state in states.iter() {
                if let Some(ts) = self.transition.get(&(state, c)) {
                    next.union_with(ts);
                }
            }
            // println!(" -> {:?}", next);
            std::mem::swap(&mut states, &mut next);
        }

        states.into_iter().any(|s| self.finals.contains(&s))
    }


    pub fn matches_rec(&self, s: &str) -> bool {
        fn try_match(nfa: &NFA, mut state: usize, mut iter: std::str::Chars, lvl: usize) -> bool {
            while let Some(c) = iter.next() {
                let t = nfa.transition.get(&(state, c));
                // print!("{0:1$}", "", lvl);
                // println!("state: {:?}; char: {:?}; t: {:?}", state, c, t);
                for next in t.into_iter().flat_map(|x| x) {
                    // print!("{0:1$}", "", lvl);
                    // println!("try: {:?}, {:?} -> {:?}", state, c, next);
                    let matchp = try_match(nfa, next, iter.clone(), lvl+2);
                    // print!("{0:1$}", "", lvl*2);
                    // println!("matches from: {:?} -> {:?}", next, matchp);
                    if matchp {
                        return true
                    }
                    // print!("{0:1$}", "", lvl); println!("try again: {:?}, {:?}", state, c);
                }
                // print!("{0:1$}", "", lvl); println!("Out of options!");
                return false
            }
            let matchp = nfa.finals.contains(&state);
            // print!("{0:1$}", "", lvl);
            // println!("END: {:?}, final? {:?}", state, matchp);
            matchp
        }
        // println!("Matching: {:?} against {:?}", s, self);
        try_match(self, self.initial, s.chars(), 0)
    }

    pub fn matches_bt(&self, s: &str) -> bool {
        let mut pending = VecDeque::new();
        pending.push_back((self.initial, s.chars().enumerate()));

        while let Some((mut state, mut input)) = pending.pop_front() {
            while let Some((i, c)) = input.next() {
                // print!("{:?}@{:?};", (i ,c), state);
                if let Some(ts) = self.transition.get(&(state, c)) {
                    // print!("{:?}@{:?} -> {:?}; ", (i, c), state, ts);
                    let mut tsit = ts.into_iter();
                    if let Some(it) = tsit.next() {
                        state = it;

                        for t in tsit {
                            pending.push_back((t, input.clone()));
                        }

                    }
                } else {
                    // println!("{:?}@{:?} -> Empty", (i, c), state);
                    break;
                }
            }

            // println!("EOS: {:?}; endp: {:?}", state, self.finals.contains(&state));
            if self.finals.contains(&state) {
                return true
            }
        }
        // try_match(self, self.initial, s.chars(), 0)
        false
    }
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
    use super::{RcRe,NFA};
    #[test]
    fn should_build_nfa() {
        use super::RcRe as R;
        let re = R::lit('a') * R::lit('b') * R::lit('a') * R::lit('b') +
                R::lit('a') * R::lit('b') * R::lit('b') * R::lit('b');
//         println!("Re: {:?}", re);
        let nfa = NFA::build(&re);
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
        let nfa = NFA::build(&re);
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
        let nfa = NFA::build(&re);
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
        let nfa = NFA::build(&re);
//         println!("NFA: {:?}", nfa);
//         println!("---");
        assert!(nfa.matches("ae"));
        assert!(!nfa.matches("e"));
        assert!(!nfa.matches("ea"));

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
        let nfa = NFA::build(&re);
        println!("NFA: {:#?}", nfa);
        assert!(nfa.matches_bt(&s));
        assert!(false);
    }
}
