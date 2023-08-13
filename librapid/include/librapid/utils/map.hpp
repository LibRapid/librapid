#ifndef LIBRAPID_UTILS_MAP_HPP
#define LIBRAPID_UTILS_MAP_HPP

namespace librapid {
    template<typename Key, typename Value>
    class Map : public std::map<Key, Value> {
    public:
        /// \brief Check if a key exists in the map
        /// \param key Key to search for
        /// \return Boolean
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool contains(const Key &key) const {
            return this->find(key) != this->end();
        }

        /// \brief Check if a key exists in the map and, if it does, set \p value to the value of
        /// the key. The function returns true if the key exists, false otherwise. (If the function
        /// returns false, \p value will not be modified/initialized, so make sure you check the
        /// return value!)
        /// \param key Key to search for
        /// \param value Value of the key, if it exists (output)
        /// \return True if the key exists, false otherwise
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool contains(const Key &key,
                                                                Value &value) const {
            auto it = this->find(key);
            if (it != this->end()) {
                value = it->second;
                return true;
            }
            return false;
        }

        /// \brief Get the value of a key
        /// \param key Key to search for
        /// \return Value of the key
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto get(const Key &key) const {
            return (*this)[key];
        }

        /// \brief Get the value of a key, or a default value if the key does not exist
        /// \param key Key to search for
        /// \param defaultValue Default value to return if the key does not exist
        /// \return Value of the key, or \p defaultValue if the key does not exist
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto get(const Key &key,
                                                           const Value &defaultValue) const {
            auto it = this->find(key);
            if (it != this->end()) { return it->second; }
            return defaultValue;
        }

        LIBRAPID_NODISCARD std::string str(const std::string &keyFormat   = "{}",
                                           const std::string &valueFormat = "{}") const {
            std::string str = "[\n";
            for (const auto &pair : *this) {
                str += "  " + fmt::format(keyFormat, pair.first);
                str += ": ";
                str += fmt::format(valueFormat, pair.second);
                str += "\n";
            }
            str += "]";

            return str;
        }
    };

    template<typename Key, typename Value>
    class UnorderedMap : public std::unordered_map<Key, Value> {
    public:
        /// \brief Check if a key exists in the map
        /// \param key Key to search for
        /// \return Boolean
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool contains(const Key &key) const {
            return this->find(key) != this->end();
        }

        /// \brief Check if a key exists in the map and, if it does, set \p value to the value of
        /// the key. The function returns true if the key exists, false otherwise. (If the function
        /// returns false, \p value will not be modified/initialized, so make sure you check the
        /// return value!)
        /// \param key Key to search for
        /// \param value Value of the key, if it exists (output)
        /// \return True if the key exists, false otherwise
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool contains(const Key &key,
                                                                Value &value) const {
            auto it = this->find(key);
            if (it != this->end()) {
                value = it->second;
                return true;
            }
            return false;
        }

        /// \brief Get the value of a key
        /// \param key Key to search for
        /// \return Value of the key
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto get(const Key &key) const {
            return (*this)[key];
        }

        /// \brief Get the value of a key, or a default value if the key does not exist
        /// \param key Key to search for
        /// \param defaultValue Default value to return if the key does not exist
        /// \return Value of the key, or \p defaultValue if the key does not exist
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto get(const Key &key,
                                                           const Value &defaultValue) const {
            auto it = this->find(key);
            if (it != this->end()) { return it->second; }
            return defaultValue;
        }

        LIBRAPID_NODISCARD std::string str(const std::string &keyFormat   = "{}",
                                           const std::string &valueFormat = "{}") const {
            std::string str = "[\n";
            for (const auto &pair : *this) {
                str += "  " + fmt::format(keyFormat, pair.first);
                str += ": ";
                str += fmt::format(valueFormat, pair.second);
                str += "\n";
            }
            str += "]";

            return str;
        }
    };
} // namespace librapid

LIBRAPID_SIMPLE_IO_IMPL(typename Key COMMA typename Value, librapid::Map<Key COMMA Value>)
LIBRAPID_SIMPLE_IO_NORANGE(typename Key COMMA typename Value, librapid::Map<Key COMMA Value>)
LIBRAPID_SIMPLE_IO_IMPL(typename Key COMMA typename Value, librapid::UnorderedMap<Key COMMA Value>)
LIBRAPID_SIMPLE_IO_NORANGE(typename Key COMMA typename Value,
                           librapid::UnorderedMap<Key COMMA Value>)

#endif // LIBRAPID_UTILS_MAP_HPP