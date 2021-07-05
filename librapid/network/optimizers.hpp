#ifndef LIBRAPID_OPTIMIZERS
#define LIBRAPID_OPTIMIZERS

#include <librapid/config.hpp>
#include <librapid/ndarray/ndarray.hpp>

namespace librapid
{
	namespace optimizers
	{
		template<typename T = double>
		class basic_optimizer
		{
		public:
			virtual ~basic_optimizer()
			{}

			LR_INLINE virtual basic_ndarray<T> apply(const basic_ndarray<T> &w,
													 const basic_ndarray<T> &dx) = 0;

			LR_INLINE virtual void set_param(const std::string &name, const T val)
			{};
			LR_INLINE virtual void set_param(const std::string &name, const basic_ndarray<T> &val)
			{};

			LR_INLINE virtual const basic_ndarray<T> get_param(const std::string &name) const = 0;
		};

		template<typename T = double>
		class sgd : public basic_optimizer<T>
		{
		public:
			sgd(const T learning_rate = 1e-2)
				: m_learning_rate(learning_rate == -1 ? 1e-2 : learning_rate)
			{
				if (learning_rate <= 0)
					throw std::logic_error("Learning rate of " + std::to_string(learning_rate) +
										   " will result in potentially negative learning. Please "
										   "ensure it is a positive value");
			}

			LR_INLINE basic_ndarray<T> apply(const basic_ndarray<T> &w, const basic_ndarray<T> &dw)
			{
				return w + m_learning_rate * dw;
			}

			LR_INLINE void set_param(const std::string &name, const T val) override
			{
				if (name == "learning rate")
				{
					m_learning_rate = val;
					return;
				}

				throw std::invalid_argument("'Stochastic Gradient Descent' optimizer has no "
											"parameter named '" + name + "'");
			}

			LR_INLINE void set_param(const std::string &name,
									 const basic_ndarray<T> &val) override
			{
				if (name == "learning rate")
				{
					m_learning_rate = val.to_scalar();
					return;
				}

				throw std::invalid_argument("'Stochastic Gradient Descent' optimizer has no "
											"parameter named '" + name + "'");
			}

			LR_INLINE const basic_ndarray<T> get_param(const std::string &name) const override
			{
				if (name == "learning rate")
					return from_data(m_learning_rate);

				throw std::invalid_argument("'Stochastic Gradient Descent' optimizer has no "
											"parameter named '" + name + "'");
			}

		private:
			T m_learning_rate = 1e-2;
		};

		template<typename T = double>
		class sgd_momentum : public basic_optimizer<T>
		{
		public:
			sgd_momentum(T learning_rate = 1e-2, T momentum = 0.9,
						 const basic_ndarray<T> &velocity = basic_ndarray<T>())
				: m_learning_rate(learning_rate == -1 ? 1e-2 : learning_rate),
				m_momentum(momentum), m_velocity(velocity)
			{
				if (learning_rate <= 0)
					throw std::logic_error("Learning rate of " + std::to_string(learning_rate) +
										   " will result in potentially negative learning. Please "
										   "ensure it is a positive value");
			}

			LR_INLINE basic_ndarray<T> apply(const basic_ndarray<T> &w,
											 const basic_ndarray<T> &dw) override
			{
				if (!m_velocity.is_initialized())
					m_velocity = zeros_like(w);

				// Momentum update formula -- also update velocity
				m_velocity = m_learning_rate * dw - m_momentum * m_velocity;
				return w + m_velocity;
			}

			LR_INLINE void set_param(const std::string &name, const T val) override
			{
				if (name == "learning rate")
				{
					m_learning_rate = val;
					return;
				}

				if (name == "momentum")
				{
					m_momentum = val;
					return;
				}

				if (name == "velocity")
				{
					m_velocity.fill(val);
					return;
				}

				throw std::invalid_argument("'Stochastic Gradient Descent With Momentum' "
											"optimizer has no "
											"parameter named '" + name + "'");
			}

			LR_INLINE void set_param(const std::string &name, const basic_ndarray<T> &val) override
			{
				if (name == "learning rate")
				{
					m_learning_rate = val.to_scalar();
					return;
				}

				if (name == "momentum")
				{
					m_momentum = val.to_scalar();
					return;
				}

				if (name == "velocity")
				{
					m_velocity = val.to_scalar();
					return;
				}

				throw std::invalid_argument("'Stochastic Gradient Descent With Momentum' "
											"optimizer has no "
											"parameter named '" + name + "'");
			}

			LR_INLINE const basic_ndarray<T> get_param(const std::string &name) const override
			{
				if (name == "learning rate")
					return from_data(m_learning_rate);
				if (name == "momentum")
					return from_data(m_momentum);
				if (name == "velocity")
					return m_velocity;

				throw std::invalid_argument("'Stochastic Gradient Descent With Momentum' "
											"optimizer has no "
											"parameter named '" + name + "'");
			}

		private:
			T m_learning_rate = 1e-2;
			T m_momentum = 0.9;
			basic_ndarray<T> m_velocity;
		};

		template<typename T = double>
		class rmsprop : public basic_optimizer<T>
		{
		public:
			rmsprop(T learning_rate = 1e-2, T decay_rate = 0.99, T epsilon = 1e-8,
					const basic_ndarray<T> &cache = basic_ndarray<T>())
				: m_learning_rate(learning_rate == -1 ? 1e-2 : learning_rate),
				m_decay_rate(decay_rate), m_epsilon(epsilon), m_cache(cache)
			{
				if (learning_rate <= 0)
					throw std::logic_error("Learning rate of " + std::to_string(learning_rate) +
										   " will result in potentially negative learning. Please "
										   "ensure it is a positive value");
			}

			LR_INLINE basic_ndarray<T> apply(const basic_ndarray<T> &x,
											 const basic_ndarray<T> &dx) override
			{
				if (!m_cache.is_initialized())
					m_cache.set_to(zeros_like(x));

				m_cache.set_to(m_decay_rate * m_cache + ((T) 1 - m_decay_rate) * (dx * dx));
				auto next_x = x + (m_learning_rate * dx) / (sqrt(m_cache) + m_epsilon);

				return next_x;
			}

			LR_INLINE void set_param(const std::string &name, const T val) override
			{
				if (name == "learning rate")
				{
					m_learning_rate = val;
					return;
				}

				if (name == "decay rate")
				{
					m_decay_rate = val;
					return;
				}

				if (name == "epsilon")
				{
					m_epsilon = val;
					return;
				}

				if (name == "cache")
				{
					m_cache.fill(val);
					return;
				}

				throw std::invalid_argument("'RMS Prop' optimizer has no "
											"parameter named '" + name + "'");
			}

			LR_INLINE void set_param(const std::string &name, const basic_ndarray<T> &val) override
			{
				if (name == "learning rate")
				{
					m_learning_rate = val.to_scalar();
					return;
				}

				if (name == "decay rate")
				{
					m_decay_rate = val.to_scalar();
					return;
				}

				if (name == "epsilon")
				{
					m_epsilon = val.to_scalar();
					return;
				}

				if (name == "cache")
				{
					m_cache = val;
					return;
				}

				throw std::invalid_argument("'RMS Prop' optimizer has no "
											"parameter named '" + name + "'");
			}

			LR_INLINE const basic_ndarray<T> get_param(const std::string &name) const override
			{
				if (name == "learning rate")
					return from_data(m_learning_rate);
				if (name == "decay rate")
					return from_data(m_decay_rate);
				if (name == "m_Epsilon")
					return from_data(m_epsilon);
				if (name == "cache")
					return m_cache;

				throw std::invalid_argument("'RMS Prop' optimizer has no "
											"parameter named '" + name + "'");
			}

		private:
			T m_learning_rate = 1e-2;
			T m_decay_rate = 0.99;
			T m_epsilon = 1e-8;
			basic_ndarray<T> m_cache;
		};

		template<typename T = double>
		class adam : public basic_optimizer<T>
		{
		public:
			adam(T learning_rate = 1e-3, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8,
				 const basic_ndarray<T> &m = basic_ndarray<T>(),
				 const basic_ndarray<T> &v = basic_ndarray<T>(), lr_int time = 0)
				: m_learning_rate(learning_rate == -1 ? 1e-3 : learning_rate),
				m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon), m_m(m), m_v(v), m_time(time)
			{
				if (learning_rate <= 0)
					throw std::logic_error("Learning rate of " + std::to_string(learning_rate) +
										   " will result in potentially negative learning. Please "
										   "ensure it is a positive value");
			}

			LR_INLINE basic_ndarray<T> apply(const basic_ndarray<T> &x,
											 const basic_ndarray<T> &dx) override
			{
				if (!m_m.is_initialized())
					m_m = zeros_like(x);

				if (!m_v.is_initialized())
					m_v = zeros_like(x);

				m_time++;
				m_m = m_beta1 * m_m + ((T) 1 - m_beta1) * dx;
				auto corr_m = m_m / ((T) 1 - std::pow(m_beta1, (T) m_time));
				m_v = m_beta2 * m_v + ((T) 1 - m_beta2) * (dx * dx);
				auto corr_v = m_v / ((T) 1 - std::pow(m_beta2, (T) m_time));
				auto next_x = x + m_learning_rate * corr_m / (sqrt(corr_v) + m_epsilon);

				return next_x;
			}

			LR_INLINE void set_param(const std::string &name, const T val) override
			{
				if (name == "learning rate")
				{
					m_learning_rate = val;
					return;
				}

				if (name == "beta1")
				{
					m_beta1 = val;
					return;
				}

				if (name == "beta2")
				{
					m_beta2 = val;
					return;
				}

				if (name == "epsilon")
				{
					m_epsilon = val;
					return;
				}

				if (name == "m")
				{
					m_m.fill(val);
					return;
				}

				if (name == "v")
				{
					m_v.fill(val);
					return;
				}

				if (name == "time")
				{
					m_time = (lr_int) val;
					return;
				}

				throw std::invalid_argument("'ADAM' optimizer has no "
											"parameter named '" + name + "'");
			}

			LR_INLINE void set_param(const std::string &name, const basic_ndarray<T> &val) override
			{
				if (name == "learning rate")
				{
					m_learning_rate = val.to_scalar();
					return;
				}

				if (name == "beta1")
				{
					m_beta1 = val.to_scalar();
					return;
				}

				if (name == "beta2")
				{
					m_beta2 = val.to_scalar();
					return;
				}

				if (name == "epsilon")
				{
					m_epsilon = val.to_scalar();
					return;
				}

				if (name == "m")
				{
					m_m = val;
					return;
				}

				if (name == "v")
				{
					m_v = val;
					return;
				}

				if (name == "time")
				{
					m_time = (lr_int) val.to_scalar();
					return;
				}

				throw std::invalid_argument("'ADAM' optimizer has no "
											"parameter named '" + name + "'");
			}

			LR_INLINE const basic_ndarray<T> get_param(const std::string &name) const override
			{
				if (name == "learning rate")
					return from_data(m_learning_rate);
				if (name == "beta1")
					return from_data(m_beta1);
				if (name == "beta2")
					return from_data(m_beta2);
				if (name == "epsilon")
					return from_data(m_epsilon);
				if (name == "m")
					return m_m;
				if (name == "v")
					return m_v;
				if (name == "time")
					return from_data((T) m_time);

				throw std::invalid_argument("'ADAM' optimizer has no "
											"parameter named '" + name + "'");
			}

		private:
			T m_learning_rate = 1e-3;
			T m_beta1 = 0.9;
			T m_beta2 = 0.999;
			T m_epsilon = 1e-8;
			basic_ndarray<T> m_m;
			basic_ndarray<T> m_v;
			lr_int m_time = 0;
		};
	}
}

#endif // LIBRAPID_OPTIMIZERS