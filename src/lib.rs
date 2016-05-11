// http://semantic-domain.blogspot.co.uk/2013/11/antimirov-derivatives-for-regular.html

use std::collections::{BTreeMap,BTreeSet, VecDeque};
use std::rc::Rc;
use std::ops::BitOr;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Re {
    Byte(u8),
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

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct RcRe(Rc<Re>);
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Res(BTreeSet<RcRe>);
// type M<Y> = BTreeMap<R, Y>;

impl Res {
    fn new() -> Res {
        Res(BTreeSet::new())
    }

    fn rmap<F: Fn(&RcRe) -> RcRe>(&self, f: F) -> Res {
        Res(self.0.iter().map(f).collect())
    }

    fn deriv(&self, c: u8) -> Res {
        Res(self.0.iter().flat_map(|r| r.deriv(c).0.into_iter()).collect())
    }

    fn is_null(&self) -> bool {
        self.0.iter().any(|r| r.is_null())
    }
}

impl<'a> BitOr for &'a Res {
    type Output = Res;
    fn bitor(self, other:Self) -> Res {
        Res(&self.0 | &other.0)
    }
}

impl From<BTreeSet<RcRe>> for Res {
    fn from(s: BTreeSet<RcRe>) -> Res {
        Res(s)
    }
}

impl RcRe {
    fn is_null(&self) -> bool {
        match &*self.0 {
            &Re::Byte(_) | &Re::Bot => false,
            &Re::Nil | &Re::Star(_) => true,
            &Re::Alt(ref l, ref r) => l.is_null() || r.is_null(),
            &Re::Seq(ref l, ref r) => l.is_null() && r.is_null(),
        }
    }
    fn nil() -> RcRe {
        RcRe(Rc::new(Re::Nil))
    }
    fn lit(c: u8) -> RcRe {
        RcRe(Rc::new(Re::Byte(c)))
    }
    fn seq(l: RcRe, r: RcRe) -> RcRe {
        RcRe(Rc::new(Re::Seq(l, r)))
    }
    fn star(r: RcRe) -> RcRe {
        RcRe(Rc::new(Re::Star(r)))
    }
    
    /*
let rec deriv c = function
  | C c' when c = c' -> R.singleton Nil 
  | C _ | Nil | Bot -> R.empty
  | Alt(r, r') -> R.union (deriv c r) (deriv c r')
  | Seq(r1, r2) -> R.union (rmap (fun r1' -> Seq(r1', r2)) (deriv c r1))
                           (if null r1 then deriv c r2 else R.empty)
  | Star r -> rmap (fun r' -> Seq(r', Star r)) (deriv c r)
       */
    fn deriv(&self, c:u8) -> Res {
        match &*self.0 {
            &Re::Byte(m) if c == m => Res(vec![RcRe::nil()].into_iter().collect()),
            &Re::Byte(_) | &Re::Nil | &Re::Bot => Res::new(),
            &Re::Alt(ref l, ref r) => &l.deriv(c) | &r.deriv(c),
            &Re::Seq(ref l, ref r) => 
                &l.deriv(c).rmap(|l2| RcRe::seq(l2.clone(), r.clone()))
                | &if l.is_null() { r.deriv(c) } else { Res::new() },
            &Re::Star(ref r) => r.deriv(c).rmap(|r2| RcRe::seq(r2.clone(), RcRe::star(r.clone()))),
        }
    }
}


#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
struct DFA {
    size: usize,
    fail: usize,
    trans: Vec<(usize, u8, usize)>,
    finalp: BTreeSet<usize>,
}

fn charfold<F: Fn(V, u8) -> V, V>(f: F, zero: V) -> V {
    (0u8..256).fold(zero, f)
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

#[cfg(test)]
mod tests {
    use super::{RcRe,Res};
    #[test]
    fn it_works() {
        let re = RcRe::star(RcRe::lit(b'a'));
        println!("{:?} -> {:?}", re, re.is_null());
        assert!(re.is_null());
        let deriv1 = re.deriv(b'a');
        println!("{:?} / {:?} -> {:?}; {:?}", re, "a", deriv1, deriv1.is_null());
        assert!(deriv1.is_null());
        let deriv2 = deriv1.deriv(b'a');
        println!("{:?} / {:?} -> {:?}; {:?}", re, "aa", deriv2, deriv2.is_null());
        assert!(deriv2.is_null());
    }
}
