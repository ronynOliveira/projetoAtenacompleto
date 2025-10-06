#!/usr/bin/env python3
"""
Script de Diagnóstico para Erro de Importação - Atena Debug
Criado para diagnosticar problemas de importação no servidor unificado

Uso:
    python debug_import_error.py
    
Ou dentro do container Docker:
    docker-compose run --rm atena-unified-server-1 python debug_import_error.py
"""

import sys
import os
import traceback
import importlib.util
import ast
import subprocess
from pathlib import Path
import json

class AtenaDebugger:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.success = []
        self.current_dir = os.getcwd()
        
    def log_error(self, test_name, error):
        self.errors.append({
            'test': test_name,
            'error': str(error),
            'type': type(error).__name__
        })
        print(f"❌ {test_name}: {error}")
        
    def log_warning(self, test_name, warning):
        self.warnings.append({
            'test': test_name,
            'warning': str(warning)
        })
        print(f"⚠️  {test_name}: {warning}")
        
    def log_success(self, test_name, message="OK"):
        self.success.append({
            'test': test_name,
            'message': message
        })
        print(f"✅ {test_name}: {message}")

    def print_separator(self, title):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    def test_environment(self):
        """Testa o ambiente Python básico"""
        self.print_separator("TESTE 1: AMBIENTE PYTHON")
        
        print(f"Python Version: {sys.version}")
        print(f"Python Executable: {sys.executable}")
        print(f"Current Directory: {self.current_dir}")
        print(f"Python Path: {sys.path}")
        
        # Verificar se estamos em Docker
        try:
            with open('/proc/1/cgroup', 'r') as f:
                if 'docker' in f.read():
                    self.log_success("Environment", "Executando dentro do Docker")
                else:
                    self.log_warning("Environment", "Não parece estar em Docker")
        except:
            self.log_warning("Environment", "Não foi possível determinar se está em Docker")

    def test_file_structure(self):
        """Verifica a estrutura de arquivos"""
        self.print_separator("TESTE 2: ESTRUTURA DE ARQUIVOS")
        
        target_file = "atena_servidor_unified.py"
        
        # Verificar se o arquivo principal existe
        if os.path.exists(target_file):
            self.log_success("File Exists", f"{target_file} encontrado")
            
            # Verificar tamanho do arquivo
            size = os.path.getsize(target_file)
            self.log_success("File Size", f"{size} bytes")
            
            # Verificar permissões
            if os.access(target_file, os.R_OK):
                self.log_success("File Permissions", "Arquivo legível")
            else:
                self.log_error("File Permissions", "Arquivo não legível")
                
        else:
            self.log_error("File Exists", f"{target_file} não encontrado")
            
        # Listar arquivos Python no diretório
        py_files = list(Path('.').glob('*.py'))
        print(f"\nArquivos Python encontrados: {len(py_files)}")
        for f in py_files:
            print(f"  - {f}")

    def test_syntax(self):
        """Verifica sintaxe de arquivos Python"""
        self.print_separator("TESTE 3: VERIFICAÇÃO DE SINTAXE")
        
        target_file = "atena_servidor_unified.py"
        
        if not os.path.exists(target_file):
            self.log_error("Syntax Check", f"{target_file} não encontrado")
            return
            
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                source = f.read()
                
            # Tentar compilar o código
            compile(source, target_file, 'exec')
            self.log_success("Syntax Check", f"{target_file} tem sintaxe válida")
            
            # Verificar usando AST
            try:
                ast.parse(source)
                self.log_success("AST Parse", "Estrutura AST válida")
            except SyntaxError as e:
                self.log_error("AST Parse", f"Erro de sintaxe na linha {e.lineno}: {e.msg}")
            
        except UnicodeDecodeError as e:
            self.log_error("Encoding", f"Problema de encoding: {e}")
        except SyntaxError as e:
            self.log_error("Syntax Check", f"Erro de sintaxe na linha {e.lineno}: {e.msg}")
        except Exception as e:
            self.log_error("Syntax Check", f"Erro inesperado: {e}")

    def test_dependencies(self):
        """Verifica dependências Python"""
        self.print_separator("TESTE 4: VERIFICAÇÃO DE DEPENDÊNCIAS")
        
        required_modules = [
            'fastapi',
            'uvicorn', 
            'asyncio',
            'logging',
            'threading',
            'concurrent.futures',
            'dataclasses',
            'os',
            'uuid'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
                self.log_success("Dependency", f"{module} disponível")
            except ImportError as e:
                self.log_error("Dependency", f"{module} não encontrado: {e}")
            except Exception as e:
                self.log_error("Dependency", f"{module} erro inesperado: {e}")

    def test_pip_packages(self):
        """Verifica pacotes pip instalados"""
        self.print_separator("TESTE 5: PACOTES PIP")
        
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("Pacotes instalados:")
                lines = result.stdout.strip().split('\n')[2:]  # Skip header
                for line in lines[:10]:  # Show first 10
                    print(f"  {line}")
                if len(lines) > 10:
                    print(f"  ... e mais {len(lines) - 10} pacotes")
                self.log_success("Pip List", f"{len(lines)} pacotes encontrados")
            else:
                self.log_error("Pip List", f"Erro ao listar pacotes: {result.stderr}")
        except Exception as e:
            self.log_error("Pip List", f"Erro ao executar pip: {e}")

    def test_manual_import(self):
        """Testa importação manual do módulo"""
        self.print_separator("TESTE 6: IMPORTAÇÃO MANUAL")
        
        target_module = "atena_servidor_unified"
        
        # Método 1: Import direto
        try:
            module = __import__(target_module)
            self.log_success("Direct Import", f"{target_module} importado com sucesso")
            
            # Verificar atributos do módulo
            attrs = [attr for attr in dir(module) if not attr.startswith('_')]
            print(f"Atributos públicos do módulo: {attrs}")
            
            # Verificar se tem a variável 'app'
            if hasattr(module, 'app'):
                self.log_success("FastAPI App", "Variável 'app' encontrada no módulo")
                app = getattr(module, 'app')
                print(f"Tipo da app: {type(app)}")
            else:
                self.log_error("FastAPI App", "Variável 'app' não encontrada no módulo")
                
        except Exception as e:
            self.log_error("Direct Import", f"Falha na importação: {e}")
            print(f"Traceback completo:")
            traceback.print_exc()
            
        # Método 2: Import usando importlib
        try:
            spec = importlib.util.spec_from_file_location(target_module, f"{target_module}.py")
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.log_success("Importlib Import", "Importação via importlib bem-sucedida")
            else:
                self.log_error("Importlib Import", "Não foi possível criar spec do módulo")
        except Exception as e:
            self.log_error("Importlib Import", f"Falha na importação via importlib: {e}")

    def test_uvicorn_compatibility(self):
        """Testa compatibilidade com Uvicorn"""
        self.print_separator("TESTE 7: COMPATIBILIDADE UVICORN")
        
        try:
            import uvicorn
            self.log_success("Uvicorn Import", f"Uvicorn {uvicorn.__version__} disponível")
            
            # Tentar carregar a aplicação como o Uvicorn faria
            try:
                from uvicorn.importer import import_from_string
                app = import_from_string("atena_servidor_unified:app")
                self.log_success("Uvicorn Load", "Aplicação carregada pelo Uvicorn com sucesso")
                print(f"Tipo da aplicação: {type(app)}")
            except Exception as e:
                self.log_error("Uvicorn Load", f"Falha ao carregar via Uvicorn: {e}")
                
        except ImportError:
            self.log_error("Uvicorn Import", "Uvicorn não está disponível")

    def analyze_code_structure(self):
        """Analisa a estrutura do código em busca de problemas"""
        self.print_separator("TESTE 8: ANÁLISE DE ESTRUTURA")
        
        target_file = "atena_servidor_unified.py"
        
        if not os.path.exists(target_file):
            self.log_error("Code Analysis", f"{target_file} não encontrado")
            return
            
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Verificar imports comentados
            commented_imports = [line.strip() for line in content.split('\n') 
                               if line.strip().startswith('# import') or 
                                  line.strip().startswith('#import')]
            
            if commented_imports:
                self.log_warning("Commented Imports", f"{len(commented_imports)} imports comentados encontrados")
                for imp in commented_imports[:5]:  # Show first 5
                    print(f"    {imp}")
                    
            # Verificar se há código executável no nível do módulo
            try:
                tree = ast.parse(content)
                module_level_code = []
                for node in tree.body:
                    if isinstance(node, (ast.Expr, ast.Assign, ast.AugAssign)):
                        module_level_code.append(ast.get_source_segment(content, node) or str(type(node).__name__))
                        
                if module_level_code:
                    self.log_warning("Module Level Code", f"{len(module_level_code)} linhas de código no nível do módulo")
                    
            except Exception as e:
                self.log_error("AST Analysis", f"Erro na análise AST: {e}")
                
        except Exception as e:
            self.log_error("Code Analysis", f"Erro na análise: {e}")

    def test_docker_specific(self):
        """Testes específicos para ambiente Docker"""
        self.print_separator("TESTE 9: ESPECÍFICOS DO DOCKER")
        
        # Verificar variáveis de ambiente
        important_env_vars = ['PYTHONPATH', 'PATH', 'WORKDIR', 'USER']
        for var in important_env_vars:
            value = os.environ.get(var)
            if value:
                print(f"{var}: {value}")
            else:
                self.log_warning("Environment", f"{var} não definida")
                
        # Verificar se estamos no diretório correto
        expected_files = ['atena_servidor_unified.py', 'requirements.txt', 'Dockerfile']
        found_files = [f for f in expected_files if os.path.exists(f)]
        
        if len(found_files) == len(expected_files):
            self.log_success("Docker Structure", "Todos os arquivos esperados encontrados")
        else:
            missing = set(expected_files) - set(found_files)
            self.log_warning("Docker Structure", f"Arquivos faltando: {missing}")

    def generate_report(self):
        """Gera relatório final"""
        self.print_separator("RELATÓRIO FINAL")
        
        total_tests = len(self.success) + len(self.warnings) + len(self.errors)
        
        print(f"Total de testes executados: {total_tests}")
        print(f"✅ Sucessos: {len(self.success)}")
        print(f"⚠️  Avisos: {len(self.warnings)}")
        print(f"❌ Erros: {len(self.errors)}")
        
        if self.errors:
            print(f"\n🔍 ERROS CRÍTICOS ENCONTRADOS:")
            for error in self.errors:
                print(f"  • {error['test']}: {error['error']}")
                
        if self.warnings:
            print(f"\n⚠️  AVISOS:")
            for warning in self.warnings:
                print(f"  • {warning['test']}: {warning['warning']}")
                
        # Recomendações
        print(f"\n💡 RECOMENDAÇÕES:")
        
        if any('Syntax' in error['test'] for error in self.errors):
            print("  1. Corrija os erros de sintaxe encontrados")
            
        if any('Dependency' in error['test'] for error in self.errors):
            print("  2. Instale as dependências faltantes usando pip")
            
        if any('Import' in error['test'] for error in self.errors):
            print("  3. Verifique a estrutura de módulos e imports")
            
        if any('Uvicorn' in error['test'] for error in self.errors):
            print("  4. Verifique a compatibilidade com Uvicorn")
            
        # Salvar relatório em arquivo
        report_data = {
            'success': self.success,
            'warnings': self.warnings,
            'errors': self.errors,
            'environment': {
                'python_version': sys.version,
                'current_dir': self.current_dir,
                'python_path': sys.path
            }
        }
        
        try:
            with open('debug_report.json', 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\n📄 Relatório detalhado salvo em: debug_report.json")
        except Exception as e:
            print(f"\n❌ Erro ao salvar relatório: {e}")

    def run_all_tests(self):
        """Executa todos os testes"""
        print("🔍 INICIANDO DIAGNÓSTICO DE ERRO DE IMPORTAÇÃO - ATENA")
        print("=" * 60)
        
        self.test_environment()
        self.test_file_structure()
        self.test_syntax()
        self.test_dependencies()
        self.test_pip_packages()
        self.test_manual_import()
        self.test_uvicorn_compatibility()
        self.analyze_code_structure()
        self.test_docker_specific()
        self.generate_report()

if __name__ == "__main__":
    debugger = AtenaDebugger()
    debugger.run_all_tests()