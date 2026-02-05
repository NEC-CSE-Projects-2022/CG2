import { Link } from 'react-router-dom';
import { HeartPulse, Github, Twitter, Linkedin } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  const footerLinks = [
    { name: 'Home', path: '/' },
    { name: 'Risk Assessment', path: '/predict' },
    { name: 'Results', path: '/results' },
    { name: 'About', path: '/about' },
    { name: 'Privacy Policy', path: '/privacy' },
    { name: 'Terms of Service', path: '/terms' },
  ];

  const socialLinks = [
    {
      name: 'GitHub',
      icon: <Github className="h-5 w-5" />,
      href: 'https://github.com/yourusername/cardiorisk-ai',
    },
    {
      name: 'Twitter',
      icon: <Twitter className="h-5 w-5" />,
      href: 'https://twitter.com/yourusername',
    },
    {
      name: 'LinkedIn',
      icon: <Linkedin className="h-5 w-5" />,
      href: 'https://linkedin.com/in/yourusername',
    },
  ];

  return (
    <footer className="bg-gradient-to-r from-blue-800 to-blue-700 text-white">
      <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand Info */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <HeartPulse className="h-8 w-8 text-white" />
              <span className="text-2xl font-bold bg-gradient-to-r from-white to-blue-100 bg-clip-text text-transparent">
                CardioRisk AI
              </span>
            </div>
            <p className="text-white text-sm">
              Empowering individuals with AI-driven cardiovascular risk assessment for better heart health.
            </p>
            <div className="flex space-x-4">
              {socialLinks.map((item) => (
                <a
                  key={item.name}
                  href={item.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-white hover:text-blue-100 transition-colors"
                  aria-label={item.name}
                >
                  <span className="sr-only">{item.name}</span>
                  {item.icon}
                </a>
              ))}
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Quick Links</h3>
            <div className="mt-4 space-y-2">
              {footerLinks.slice(0, 4).map((item) => (
                <Link
                  key={item.name}
                  to={item.path}
                  className="text-white hover:text-blue-100 text-sm block transition-colors"
                >
                  {item.name}
                </Link>
              ))}
            </div>
          </div>

          {/* Resources */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Resources</h3>
            <ul className="space-y-2">
              <li><a href="#faq" className="text-white hover:text-blue-100 transition">FAQ</a></li>
              <li><a href="#privacy" className="text-white hover:text-blue-100 transition">Privacy Policy</a></li>
              <li><a href="#terms" className="text-white hover:text-blue-100 transition">Terms of Service</a></li>
            </ul>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-4">Contact Us</h3>
            <ul className="space-y-2 text-white">
              <li>Email: info@cardioguard.com</li>
              <li>Phone: (123) 456-7890</li>
              <li>Address: 123 Health St, Medical City</li>
            </ul>
          </div>
        </div>
        
        <div className="border-t border-gray-300/20 mt-8 pt-6 text-center text-white/80 text-sm">
          <p>© {new Date().getFullYear()} CardioGuard. All rights reserved.</p>
          <p className="mt-2 text-xs text-white/70">
            This tool is for informational purposes only and is not a substitute for professional medical advice.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;