#pragma once

#include "Cp2Lt9_Case.h"
#include <string>
#include <map>

/**
*/
class Cp2Lt9_Func {
public:
    struct      CaseScore        { Cp2Lt9_Case code; double score; };
    using       CaseMap          = std::map<unsigned,CaseScore>;

    /***/       Cp2Lt9_Func      ( OptParm &opt_parm, std::string float_type, std::string simd_type, int max_nb_nodes = 9 );
    /***/       Cp2Lt9_Func      ();

    int         max_log_simd_size() const;
    void        write_def        ( std::ostream &os ) const;
    double      score            () const;

    void        make_best_score  ( double &best_score, std::size_t &best_case, const std::vector<Cp2Lt9_Case> &cases, unsigned code, int nb_nodes );
    void        make_case_map    ();
    void        write_def        ( std::ostream &os, const CaseMap &case_map, std::string func_name, bool for_1_case = false ) const;

    // constraints
    int         min_nb_nodes;
    int         max_nb_nodes;
    std::string float_type;
    std::string simd_type;

    // parameters (from opt_parm)
    int         size_for_tests;  ///<
    int         nb_registers;    ///<
    bool        pi_in_regs;      ///<
    int         simd_size;       ///< size of the registers
    bool        make_di;         ///< true to make di_xx direclty (instead of bi_xx > cs, and di = bi_xx - cs later)

    //
    CaseMap     case_map;        ///< case id => best code
};

