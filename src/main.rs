extern crate antimirov;
use std::env;
use antimirov::RcRe as R;

fn main() {
        let re = R::lit('a') * R::lit('b') * R::lit('a') * R::lit('b') +
                R::lit('a') * R::lit('b') * R::lit('b') * R::lit('b');
        println!("Re: {:?}", re);
        let nfa = re.make_nfa();
        println!("NFA: {:?}", nfa);
        println!("---");

    for arg in env::args().skip(1) {
        println!("{:?}: {:?}", arg, nfa.matches(&arg));
    }
}
