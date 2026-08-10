// 출처: LT-mapper(ltslam) — https://github.com/gisbi-kim/lt-mapper
//   원본: lt-mapper-main/ltslam/include/ltslam/BetweenFactorWithAnchoring.h
//
// GTSAM 4.3.0 API 어댑트 (2026-08-10):
//   원본은 boost 기반 구 GTSAM API(boost::optional<Matrix&>, boost::shared_ptr,
//   boost::static_pointer_cast, NoiseModelFactor4)로 작성되어 GTSAM 4.3에서는
//   그대로 컴파일되지 않는다. 이 환경(GTSAM 4.3.0)에서 확인한 실제 API에 맞춰
//   다음을 변경했다:
//     - NonlinearFactor::shared_ptr 은 std::shared_ptr (boost::shared_ptr 아님)
//     - evaluateError의 옵셔널 자코비안 인자 타입은 gtsam::Matrix*
//       (OptionalMatrixType), 기본값은 nullptr (boost::none 아님)
//     - clone()의 boost::static_pointer_cast → std::static_pointer_cast
//     - NoiseModelFactor4<...>는 NoiseModelFactorN<...>의 매크로로 존재하므로
//       그대로 사용 가능 (4-key 팩터)
//     - print/equals/evaluateError/clone 시그니처에 override를 명시하여
//       베이스와의 시그니처 불일치를 컴파일 타임에 검증한다
//   수식(앵커 합성 h(x)=anchor∘local, between(hx1,hx2)) 자체는 원본과 동일하다.
#pragma once

#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

namespace gtsam {

  /**
   * A class for a measurement predicted by "between(anchor1*p1, anchor2*p2)"
   * — 세션 앵커(anchor_key1/2)를 노드 값에 미리 곱해 넣지 않고 별도 그래프
   * 변수로 명시하는 LT-mapper 방식(Form A). 논문 LT-mapper의 다중 세션
   * 포즈그래프 정식화(앵커 노드를 통한 세션-전역 좌표 변환)를 구현한다.
   * @tparam VALUE the Value type
   * @addtogroup SLAM
   */
  template<class VALUE>
  class BetweenFactorWithAnchoring: public NoiseModelFactor4<VALUE, VALUE, VALUE, VALUE> {

    // Check that VALUE type is a testable Lie group
    BOOST_CONCEPT_ASSERT((IsTestable<VALUE>));
    BOOST_CONCEPT_ASSERT((IsLieGroup<VALUE>));

  public:

    typedef VALUE T;

  private:

    typedef BetweenFactorWithAnchoring<VALUE> This;
    typedef NoiseModelFactor4<VALUE, VALUE, VALUE, VALUE> Base;

    VALUE measured_; /** The measurement */

  public:

    // shorthand for a smart pointer to a factor
    // GTSAM 4.3: NonlinearFactor::shared_ptr는 std::shared_ptr이다
    typedef typename std::shared_ptr<BetweenFactorWithAnchoring> shared_ptr;

    /** default constructor - only use for serialization */
    BetweenFactorWithAnchoring() {}

    /** Constructor */
    BetweenFactorWithAnchoring(
          // 1 for robot 1, and 2 for robot 2
          Key key1, Key key2, Key anchor_key1, Key anchor_key2,
          const VALUE& measured,
          const SharedNoiseModel& model = nullptr) :
      Base(model, key1, key2, anchor_key1, anchor_key2), measured_(measured) {
    }

    virtual ~BetweenFactorWithAnchoring() {}

    /// @return a deep copy of this factor
    gtsam::NonlinearFactor::shared_ptr clone() const override {
      return std::static_pointer_cast<gtsam::NonlinearFactor>(
          gtsam::NonlinearFactor::shared_ptr(new This(*this))); }

    /** implement functions needed for Testable */

    /** print */
    void print(const std::string& s, const KeyFormatter& keyFormatter = DefaultKeyFormatter) const override {
      std::cout << s << "BetweenFactorWithAnchoring("
          << keyFormatter(this->key1()) << ","
          << keyFormatter(this->key2()) << ")\n";
      traits<T>::Print(measured_, "  measured: ");
      this->noiseModel_->print("  noise model: ");
    }

    /** equals */
    bool equals(const NonlinearFactor& expected, double tol=1e-9) const override {
      const This *e =  dynamic_cast<const This*> (&expected);
      return e != nullptr && Base::equals(*e, tol) && traits<T>::Equals(this->measured_, e->measured_, tol);
    }

    /** implement functions needed to derive from Factor */

    // some useful link, giseop
    // line 384, https://gtsam.org/doxygen/a00317_source.html
    // https://gtsam.org/doxygen/a02091.html
    // betweenfactor https://gtsam.org/doxygen/a00935_source.html
    // line 224 https://gtsam.org/doxygen/a00053_source.html
    // isam ver. line 233, https://people.csail.mit.edu/kaess/isam/doc/slam2d_8h_source.html
    /** vector of errors */
    Vector evaluateError(
        const T& p1, const T& p2, const T& anchor_p1, const T& anchor_p2,
        gtsam::Matrix* H1 = nullptr,
        gtsam::Matrix* H2 = nullptr,
        gtsam::Matrix* anchor_H1 = nullptr,
        gtsam::Matrix* anchor_H2 = nullptr
      ) const override {

      // anchor node h(.) (ref: isam ver. line 233, https://people.csail.mit.edu/kaess/isam/doc/slam2d_8h_source.html)
      // gtsam::Matrix* → OptionalJacobian<6,6> 암시적 변환이 있어 타입만 바꾸면 그대로 동작한다
      T hx1 = traits<T>::Compose(anchor_p1, p1, anchor_H1, H1); // for the updated jacobian, see line 60, 219, https://gtsam.org/doxygen/a00053_source.html
      T hx2 = traits<T>::Compose(anchor_p2, p2, anchor_H2, H2);
      T hx = traits<T>::Between(hx1, hx2, H1, H2);

      return traits<T>::Local(measured_, hx);
    }

    /** return the measured */
    const VALUE& measured() const {
      return measured_;
    }

    /** number of variables attached to this factor */
    std::size_t size() const {
      return 4;
    }

  private:

    /** Serialization function */
    friend class boost::serialization::access;
    template<class ARCHIVE>
    void serialize(ARCHIVE & ar, const unsigned int /*version*/) {
      ar & boost::serialization::make_nvp("NoiseModelFactor4",
          boost::serialization::base_object<Base>(*this));
      ar & BOOST_SERIALIZATION_NVP(measured_);
    }

  }; // \class BetweenFactorWithAnchoring


  // traits
  template<class VALUE>
  struct traits<BetweenFactorWithAnchoring<VALUE> > : public Testable<BetweenFactorWithAnchoring<VALUE> > {};

} /// namespace gtsam
