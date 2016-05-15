#![feature(test)]
extern crate test;
extern crate antimirov;
extern crate regex;

use antimirov::{NFA, RcRe as R};
fn test_re() -> R {
    R::lit('a') * R::lit('b') * R::lit('a') * R::lit('b') +
    R::lit('a') * R::lit('b') * R::lit('b') * R::lit('b')
}


mod antimorov {
    use antimirov::{NFA, RcRe as R};
    use test;
    #[bench]
    fn non_matching(b: &mut test::Bencher) {
        let re = super::test_re();

        let nfa = NFA::build(&re);
        let arg = "abba";
        b.bytes = arg.len() as u64;
        assert!(!nfa.matches(&arg));
        b.iter(|| nfa.matches(&arg))
    }

    #[bench]
    fn matching_ok(b: &mut test::Bencher) {
        let re = super::test_re();

        let nfa = NFA::build(&re);
        let arg = "abab";
        b.bytes = arg.len() as u64;
        assert!(nfa.matches(&arg));
        b.iter(|| nfa.matches(&arg))
    }
}

mod antimorov_rec {
    use antimirov::{NFA, RcRe as R};
    use test;
    #[bench]
    fn non_matching(b: &mut test::Bencher) {
        let re = super::test_re();

        let nfa = NFA::build(&re);
        let arg = "abba";
        b.bytes = arg.len() as u64;
        assert!(!nfa.matches_rec(&arg));
        b.iter(|| nfa.matches_rec(&arg))
    }

    #[bench]
    fn matching_ok(b: &mut test::Bencher) {
        let re = super::test_re();

        let nfa = NFA::build(&re);
        let arg = "abab";
        b.bytes = arg.len() as u64;
        assert!(nfa.matches_rec(&arg));
        b.iter(|| nfa.matches_rec(&arg))
    }
}

mod antimorov_bt {
    use antimirov::{NFA, RcRe as R};
    use test;
    #[bench]
    fn non_matching(b: &mut test::Bencher) {
        let re = super::test_re();

        let nfa = NFA::build(&re);
        let arg = "abba";
        b.bytes = arg.len() as u64;
        assert!(!nfa.matches_bt(&arg));
        b.iter(|| nfa.matches_bt(&arg))
    }

    #[bench]
    fn matching_ok(b: &mut test::Bencher) {
        let re = super::test_re();

        let nfa = NFA::build(&re);
        let arg = "abab";
        b.bytes = arg.len() as u64;
        assert!(nfa.matches_bt(&arg));
        b.iter(|| nfa.matches_bt(&arg))
    }
}


mod rust_regex {
    use regex::Regex;
    use test;
    #[bench]
    fn non_matching(b: &mut test::Bencher) {
        let re = Regex::new(r"^(abab|abbb)$").unwrap();

        let arg = "abba";
        b.bytes = arg.len() as u64;
        assert!(!re.is_match(&arg));
        b.iter(|| re.is_match(&arg))
    }

    #[bench]
    fn matching_ok(b: &mut test::Bencher) {
        let re = Regex::new(r"^(abab|abbb)$").unwrap();

        let arg = "abab";
        b.bytes = arg.len() as u64;
        assert!(re.is_match(&arg));
        b.iter(|| re.is_match(&arg))
    }
}
