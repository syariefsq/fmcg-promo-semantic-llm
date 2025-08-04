"""
Simple test script for the FMCG Promotion Recommender

This is a straightforward test without complex syntax.
Easy to understand and follow for learning purposes.
"""

def test_data_loading():
    """Test if we can load the data properly."""
    print("Testing data loading...")
    
    try:
        from data import DataCleaner
        
        cleaner = DataCleaner()
        promos, skus = cleaner.clean_all_data("data/promos.csv", "data/sku_master.csv")
        
        print(f"✅ Loaded {len(promos)} promotions")
        print(f"✅ Loaded {len(skus)} SKUs")
        print(f"✅ Found {promos['category'].nunique()} categories")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_search():
    """Test if search functionality works."""
    print("\\nTesting search...")
    
    try:
        from app import PromotionApp
        
        app = PromotionApp()
        app = app.setup()
        
        if app and app.is_ready:
            # Try a simple search with the app's search function
            from app import search_promotions
            results, skus = search_promotions("DISKON 50%", {'n_results': 3})
            
            print(f"✅ Search returned {len(results)} results")
            if results:
                print(f"✅ Top result: {results[0]['headline']}")
            
            return True
        else:
            print("❌ App setup failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_enhanced_data():
    """Test if enhanced data is working."""
    print("\\nTesting enhanced data...")
    
    try:
        import pandas as pd
        
        # Load enhanced data
        enhanced = pd.read_csv("data/enhanced_promos.csv")
        print(f"✅ Enhanced data loaded: {len(enhanced)} rows")
        
        print(f"✅ Brands: {enhanced['brand'].nunique()}")
        print(f"✅ Categories: {enhanced['category'].nunique()}")
        if 'promo_type' in enhanced.columns:
            print(f"✅ Promotion types: {enhanced['promo_type'].nunique()}")
        if 'ctr' in enhanced.columns:
            print(f"✅ Enhanced metrics available (CTR, conversion rate, etc.)")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests in a simple way."""
    print("🎯 Testing FMCG Promotion Recommender")
    print("=" * 40)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Enhanced Data", test_enhanced_data),
        ("Search Function", test_search),
    ]
    
    passed = 0
    
    for test_name, test_function in tests:
        print(f"\\n🔄 {test_name}...")
        if test_function():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\\n" + "=" * 40)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! App is working!")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()