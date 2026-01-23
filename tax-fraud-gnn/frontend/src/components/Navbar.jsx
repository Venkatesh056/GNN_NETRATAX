import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import './Navbar.css'

const Navbar = () => {
  const location = useLocation()

  const navItems = [
    { path: '/', label: 'Dashboard', icon: '📊' },
    { path: '/companies', label: 'Companies', icon: '🔍' }
  ]

  const toggleTheme = () => {
    const current = document.documentElement.getAttribute('data-theme') || 'light'
    const next = current === 'dark' ? 'light' : 'dark'
    document.documentElement.setAttribute('data-theme', next)
    localStorage.setItem('netra_theme', next)
  }

  return (
    <motion.nav 
      className="navbar"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      <div className="navbar-container">
        <motion.div 
          className="navbar-brand"
          whileHover={{ scale: 1.05 }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          <h1>🚨 Tax Fraud Detection</h1>
          <p>Graph Neural Network-based Analysis</p>
        </motion.div>
        
        <ul className="navbar-menu">
          {navItems.map((item, index) => (
            <motion.li
              key={item.path}
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 + 0.3 }}
            >
              <Link
                to={item.path}
                className={`nav-link ${location.pathname === item.path ? 'active' : ''}`}
              >
                <motion.span
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {item.icon} {item.label}
                </motion.span>
              </Link>
            </motion.li>
          ))}
          <li>
            <button className="nav-link" style={{ background: 'transparent', border: 'none' }} onClick={toggleTheme} title="Toggle theme">🌗 Theme</button>
          </li>
        </ul>
      </div>
    </motion.nav>
  )
}

export default Navbar

